# This file includes modifications to https://github.com/lvwerra/trl/blob/master/trl/ppo.py” 
# distributed through the GitHub library https://github.com/lvwerra/trl/ 
# under this license https://github.com/lvwerra/trl/blob/master/LICENSE.
# Copyright with respect to the modifications: Copyright 2020 Naver Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = ['AdaptiveKLController', 'FixedKLController', 'PPOTrainer']


import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
import torch
import collections
import time
import random
import copy
import transformers
import os

from .core import (logprobs_from_logits,
                      probs_from_logits,
                         whiten,
                         clip_by_value,
                         entropy_from_logits,
                         flatten_dict,
                         average_torch_dicts,
                         stats_to_np,
                         stack_dicts,
                         add_suffix,
                         plot_grad_flow)

from transformers import (get_constant_schedule_with_warmup,
                          get_linear_schedule_with_warmup,
                          get_cosine_schedule_with_warmup,
                          get_cosine_with_hard_restarts_schedule_with_warmup)


from .base_trainer import BaseTrainer



class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """
    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult



class FixedKLController:
    """Fixed KL controller."""
    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass



class PPOTrainer(BaseTrainer):
    """
    The PPO_trainer uses Proximal Policy Optimization to optimise language models.
    """

    default_params = {
        "lr": 1.41e-5,
        "adap_kl_ctrl": True,
        "init_kl_coef":0.2,
        "target": 6,
        "horizon":10000,
        "gamma":1,
        "lam":0.95,
        "cliprange": .2,
        "cliprange_value":.2,
        "vf_coef":.1,
        "batch_size": 256,
        "forward_batch_size": 16,
        "minibatch_epochs": 4,
    }

    def __init__(self, model_cls, tokenizer, sampling_function, scoring_function, **params):

        """
        Initialize PPOTrainer.

        Args:
            model (torch.model): Hugging Face transformer GPT2 model with value head
            ref_model (torch.model): Hugging Face transformer GPT2 refrence model used for KL penalty
            params (dict or None): PPO parameters for training. Can include following keys:
                'lr' (float): Adam learning rate, default: 1.41e-5
                'batch_size' (int): Number of samples per optimisation step, default: 256
                'forward_batch_size' (int): Number of samples forward passed through model at a time, default: 16
                'ppo_epochs' (int): Number of optimisation epochs per batch of samples, default: 4
                'gamma' (float)): Gamma parameter for advantage calculation, default: 1.
                'lam' (float): Lambda parameter for advantage calcualation, default: 0.95
                'cliprange_value' (float): Range for clipping values in loss calculation, default: 0.2
                'cliprange' (float): Range for clipping in PPO policy gradient loss, default: 0.2
                'vf_coef' (float): Scaling factor for value loss, default: 0.1
                'adap_kl_ctrl' (bool): Use adaptive KL control, otherwise linear, default: True
                'init_kl_coef' (float): Initial KL penalty coefficient (used for adaptive and linear control), default: 0.2
                'target' (float): Target KL value for adaptive KL control, default: 6.0
                'horizon' (float): Horizon for adaptive KL control, default: 10000

        """
        super().__init__(tokenizer=tokenizer, sampling_function=sampling_function, scoring_function=scoring_function)

        self.params = self.default_params
        self.params.update(params)

        self.ref_model = model_cls.from_pretrained(self.params['lm_name']).to(self.params['gpt2_orig_device'])
        self.ref_model.eval()

        self.orig_model = self.ref_model
        self.model = model_cls.from_pretrained(self.params['lm_name'], attn_pdrop=self.params['dropout'],
                                            summary_first_dropout=self.params['dropout']).to(self.params['gpt2_device'])

        self.optimizer = Adam(self.model.parameters(), lr=self.params['lr'])
        self.is_policy_eval = True


        ### create LR scheduler
        scheduler_ = self.params['scheduler']
        assert scheduler_ in ['cosine', 'constant', 'linear', 'cosine_restarts'], "unknown scheduler: {}".format(self.params         ['scheduler'])
        if scheduler_ == 'constant':
            self.scheduler = get_constant_schedule_with_warmup(self.optimizer, self.params['warmup_steps'])
        elif scheduler_ == 'cosine':
            print("Cosine scheduler...")
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, self.params['warmup_steps'],
                                                            self.params['steps']//self.params['batch_size'])
        elif scheduler_ == 'linear':
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, self.params['warmup_steps'])
        elif scheduler_ == 'cosine_restarts':
            self.scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(self.optimizer,
            self.params['warmup_steps'],
            self.params['steps'] // self.params['batch_size'],
            num_cycles=self.params['num_restart_cycles'])

        self.kl_ctl = AdaptiveKLController(self.params['init_kl_coef'],
                                           self.params['target'],
                                           self.params['horizon'])

        self.iter= 0
    def get_sampling_model(self):
        return self.model

    def get_policy_model(self):
        return self.model

    def get_eval_model(self):
        return self.model

    def step(self, query, response, scores):
        """
        Run a PPO optimisation step.

        args:
            query (torch.tensor): tensor containing the encoded queries, shape [batch_size, query_length]
            response (torch.tensor): tensor containing the encoded responses, shape [batch_size, response_length]
            scores (torch.tensor): tensor containing the scores, shape [batch_size]

        returns:
            train_stats (dict): a summary of the training statistics
        """

        bs = self.params['batch_size']
        timing = dict()
        t0 = time.time()

        gen_len = response.shape[1]
        model_input = torch.cat((query, response), axis=1)

        t = time.time()
        logprobs, ref_logprobs, values = self.batched_forward_pass(model_input, gen_len)
        timing['time/ppo/forward_pass'] = time.time()-t

        t = time.time()
        rewards, non_score_reward, kl_coef = self.compute_rewards(scores, logprobs, ref_logprobs)
        timing['time/ppo/compute_rewards'] = time.time()-t

        t = time.time()
        all_stats = []

        idxs = list(range(bs))
        fbs = self.params['forward_batch_size']

        for _ in range(self.params['minibatch_epochs']):
            random.shuffle(idxs)

            for i in range(self.params['batch_size'] // fbs):
                fbs_idxs = idxs[i*fbs: (i+1)*fbs]
                train_stats = self.train_minibatch(logprobs[fbs_idxs], values[fbs_idxs],
                                                   rewards[fbs_idxs], query[fbs_idxs],
                                                   response[fbs_idxs], model_input[fbs_idxs])

                all_stats.append(train_stats)
        timing['time/ppo/optimize_step'] = time.time()-t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats['policy/advantages'] = torch.flatten(train_stats['policy/advantages']).unsqueeze(0)
        train_stats['policy/ratio'] = torch.flatten(train_stats['policy/ratio']).unsqueeze(0)

        stats = self.record_step_stats(scores=scores, logprobs=logprobs, ref_logprobs=ref_logprobs,
                                       non_score_reward=non_score_reward, train_stats=train_stats,
                                       kl_coef=kl_coef)
        stats = stats_to_np(stats)
        timing['time/ppo/calc_stats'] = time.time()-t

        self.kl_ctl.update(stats['objective/kl'], self.params['batch_size'])

        timing['time/ppo/total'] = time.time()-t0

        stats["scores"] = scores

        sample_dict = self.build_samples_buffer(self.ref_model, n_samples = self.params['batch_size'])
        Z_minibatch_mean, Z_minibatch_std = self.compute_z(sample_dict) # compute estimate of the parition function Z on minibatches

        n_previous = self.iter
        self.Z_moving_average = (Z_minibatch_mean + n_previous * self.Z_moving_average) / (n_previous + 1)


        self.iter+=1
        return stats, self.get_logs_from_stats(stats)


    def get_logs_from_stats(self, stats):
        """Method to collect stats to log / plot into tracking apps like
            Neptune or WandB
            args:
            all_stats (gdc.core.stack_dicts): all previous statistics
        """

        logs = dict()
        logs['loss'] = stats['ppo/loss/total'].mean()
        logs['lr'] = stats['ppo/lr'].mean()
        logs['KL(π||a)'] =  stats['objective/kl'] #
        logs['GPT-2 score'] = -stats['objective/ref_logprobs'].sum(axis=-1).mean() #

        logs['ppo/returns/mean'] = stats["ppo/returns/mean"].mean()
        logs['b(x)_mean'] = torch.mean(stats["scores"]).cpu().numpy()
        #logs['b(x)_std'] = np.std((stats["scores"]).cpu().numpy())
        return logs


    def batched_forward_pass(self, model_input, gen_len):
        """Calculate model outputs in multiple batches."""
        bs = self.params['batch_size']
        fbs = self.params['forward_batch_size']
        logprobs = []
        ref_logprobs = []
        values = []

        for i in range(int(self.params['batch_size']/fbs)):
            m_input = model_input[i*fbs:(i+1)*fbs]
            m_input_ref_device = m_input.to(self.params["gpt2_ref_device"])

            logits, _, v = self.model(m_input)
            ref_logits, _, _ = self.ref_model(m_input_ref_device) # move to ref device

            values.append(v[:, -gen_len-1:-1].detach())
            logprobs.append(logprobs_from_logits(logits[:,:-1,:], m_input[:,1:])[:, -gen_len:].detach())
            ref_logprobs.append(logprobs_from_logits(ref_logits[:,:-1,:], m_input_ref_device[:,1:])[:, -gen_len:].detach())

        return torch.cat(logprobs), torch.cat(ref_logprobs), torch.cat(values)

    def train_minibatch(self, logprobs, values, rewards, query, response, model_input):
        """Train one PPO minibatch"""
        loss_p, loss_v, train_stats  = self.loss(logprobs, values, rewards, query, response, model_input)
        loss = loss_p + loss_v
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        train_stats['lr'] = torch.tensor(self.scheduler.get_last_lr()) ## add learnig rate to stats
        return train_stats

    def compute_rewards(self, scores, logprobs, ref_logprobs):
        """Compute per token rewards from scores and KL-penalty."""
        kl = logprobs - ref_logprobs.to(self.params['gpt2_device'])
        non_score_reward = -self.kl_ctl.value * kl
        rewards = non_score_reward.clone().detach()
        rewards[:, -1] += scores.to(self.params['gpt2_device'])
        return rewards, non_score_reward, self.kl_ctl.value

    def loss(self, old_logprobs, values, rewards, query, response, model_input):
        """Calculate policy and value losses."""
        lastgaelam = 0
        advantages_reversed = []
        gen_len = response.shape[1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = rewards[:, t] + self.params['gamma'] * nextvalues - values[:, t]
            lastgaelam = delta + self.params['gamma'] * self.params['lam'] * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

        returns = advantages + values
        advantages = whiten(advantages)
        advantages = advantages.detach()

        logits, _, vpred = self.model(model_input)
        logprob = logprobs_from_logits(logits[:,:-1,:], model_input[:, 1:])

        #only the generation part of the values/logprobs is needed
        logprob, vpred = logprob[:, -gen_len:], vpred[:,-gen_len-1:-1]

        vpredclipped = clip_by_value(vpred,
                                     values - self.params["cliprange_value"],
                                     values + self.params["cliprange_value"])

        vf_losses1 = (vpred - returns)**2
        vf_losses2 = (vpredclipped - returns)**2
        vf_loss = .5 * torch.mean(torch.max(vf_losses1, vf_losses2))
        vf_clipfrac =  torch.mean(torch.gt(vf_losses2, vf_losses1).double())

        ratio = torch.exp(logprob - old_logprobs)

        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio,
                                               1.0 - self.params['cliprange'],
                                               1.0 + self.params['cliprange'])

        pg_loss = torch.mean(torch.max(pg_losses, pg_losses2))
        pg_clipfrac = torch.mean(torch.gt(pg_losses2, pg_losses).double())

        loss = pg_loss + self.params['vf_coef'] * vf_loss

        entropy = torch.mean(entropy_from_logits(logits))
        approxkl = .5 * torch.mean((logprob - old_logprobs)**2)
        policykl = torch.mean(logprob - old_logprobs)
        return_mean, return_var = torch.mean(returns), torch.var(returns)
        value_mean, value_var = torch.mean(values), torch.var(values)

        stats = dict(
            loss=dict(policy=pg_loss, value=vf_loss, total=loss),
            policy=dict(entropy=entropy, approxkl=approxkl,policykl=policykl, clipfrac=pg_clipfrac,
                        advantages=advantages, advantages_mean=torch.mean(advantages), ratio=ratio),
            returns=dict(mean=return_mean, var=return_var),
            val=dict(vpred=torch.mean(vpred), error=torch.mean((vpred - returns) ** 2),
                     clipfrac=vf_clipfrac, mean=value_mean, var=value_var),
        )
        return pg_loss, self.params['vf_coef'] * vf_loss, flatten_dict(stats)


    def record_step_stats(self, kl_coef, **data):
        """Record training step statistics."""
        kl = data['logprobs'] - data['ref_logprobs'].to(self.params['gpt2_device'])
        mean_kl = torch.mean(torch.sum(kl, axis=-1))
        mean_entropy = torch.mean(torch.sum(-data['logprobs'], axis=1))
        mean_non_score_reward =torch.mean(torch.sum(data['non_score_reward'], axis=1))
        stats = {

            'objective/kl': mean_kl,
            'objective/kl_dist': kl,
            'objective/logprobs': data['logprobs'],
            'objective/ref_logprobs': data['ref_logprobs'],
            'objective/kl_coef': kl_coef,
            'objective/entropy': mean_entropy,
            'ppo/mean_non_score_reward': mean_non_score_reward,
        }

        for k, v in data['train_stats'].items():
            stats[f'ppo/{k}'] = torch.mean(v, axis=0)
        stats['ppo/val/var_explained'] = 1 - stats['ppo/val/error'] / stats['ppo/returns/var']
        return stats


    def save_checkpoint(self, directory):

        ## creating checkpointting dir if not exists
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        ckpt = super().prepare_checkpoint() # prepare base checkpoint


        f_path = os.path.join(directory,'checkpoint_last.pt')
        torch.save(ckpt, f_path)

    def load_checkpoint(self, checkpoint_path):

        ckpt = torch.load(checkpoint_path)

        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.iter = ckpt['iter']



