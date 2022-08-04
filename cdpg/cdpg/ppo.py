import numpy as np
from torch.optim import Adam
import torch
import time
import random
import os

from transformers import (get_constant_schedule_with_warmup,
                          get_linear_schedule_with_warmup,
                          get_cosine_schedule_with_warmup,
                          get_cosine_with_hard_restarts_schedule_with_warmup)

from .base_trainer import BaseTrainer
from .core import logprobs_from_logits, flatten_dict, stats_to_np, stack_dicts


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

        # pi_theta policy to be learned
        self.model = model_cls.from_pretrained(params['lm_name']).to(params['gpt2_device'])

        # original model for computing kl(pi||a)
        self.orig_model = model_cls.from_pretrained(params['lm_name']).to(params['gpt2_orig_device'])

        self.ref_model = self.model

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

    def step(self, query, response, query_mask, response_mask, scores, game_data):
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

        t = time.time()
        logprobs, ref_logprobs = self.batched_forward_pass(query, response, query_mask, response_mask)
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
                train_stats = self.train_minibatch(logprobs[fbs_idxs],
                                                   rewards[fbs_idxs], query[fbs_idxs],
                                                   response[fbs_idxs], query_mask[fbs_idxs], response_mask[fbs_idxs])

                all_stats.append(train_stats)
        timing['time/ppo/optimize_step'] = time.time()-t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats['policy/ratio'] = torch.flatten(train_stats['policy/ratio']).unsqueeze(0)

        stats = self.record_step_stats(scores=scores, logprobs=logprobs, ref_logprobs=ref_logprobs,
                                       non_score_reward=non_score_reward, train_stats=train_stats,
                                       kl_coef=kl_coef)
        stats = stats_to_np(stats)
        timing['time/ppo/calc_stats'] = time.time()-t

        self.kl_ctl.update(stats['objective/kl'], self.params['batch_size'])

        timing['time/ppo/total'] = time.time()-t0

        stats["scores"] = scores

        self.iter+=1
        return stats, self.get_logs_from_stats(stats)


    def get_logs_from_stats(self, stats):
        """Method to collect stats to log / plot into tracking apps like
            Neptune or WandB
            args:
            all_stats (cdpg.core.stack_dicts): all previous statistics
        """

        logs = {}
        for k, v in stats.items():
            logs[k] = v if isinstance(v, float) else v.mean()
        logs['loss'] = stats['ppo/loss/total'].mean()
        logs['lr'] = stats['ppo/lr'].mean()
        logs['KL(pi||a)'] = stats['objective/kl'] #
        logs['GPT-2 score'] = -stats['objective/ref_logprobs'].sum(axis=-1).mean() #
        logs['b(x)_mean'] = torch.mean(stats["scores"]).cpu().numpy()
        logs['b(x)_std'] = np.std((stats["scores"]).cpu().numpy())
        return logs

    def batched_forward_pass(self, query, response, query_mask, response_mask):
        """Calculate model outputs in multiple batches."""
        fbs = self.params['forward_batch_size']
        logprobs = []
        ref_logprobs = []

        for i in range(int(self.params['batch_size']/fbs)):
            m_query = query[i*fbs:(i+1)*fbs].to(self.params["gpt2_device"])
            m_response = response[i*fbs:(i+1)*fbs].to(self.params["gpt2_device"])
            m_query_mask = query_mask[i*fbs:(i+1)*fbs].to(self.params["gpt2_device"])
            m_response_mask = response_mask[i*fbs:(i+1)*fbs].to(self.params["gpt2_device"])
            logits, _, _ = self.model(m_query, m_response, m_query_mask)
            ref_logits, _, _ = self.ref_model(m_query, m_response, m_query_mask)
            logprobs.append(logprobs_from_logits(logits[:, :-1, :], m_response[:, 1:], mask=m_response_mask[:, 1:]).detach())
            ref_logprobs.append(logprobs_from_logits(ref_logits[:, :-1, :], m_response[:, 1:], mask=m_response_mask[:, 1:]).detach())

        return torch.cat(logprobs), torch.cat(ref_logprobs)

    def train_minibatch(self, logprobs, rewards, query, response, query_mask, response_mask):
        """Train one PPO minibatch"""
        loss, train_stats = self.loss(logprobs, rewards, query, response, query_mask, response_mask)
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

    def loss(self, old_logprobs, rewards, query, response, query_mask, response_mask):
        """Calculate policy and value losses."""
        logits, _, _ = self.model(query, response, query_mask)
        logprob = logprobs_from_logits(logits[:, :-1], response[:, 1:], mask=response_mask[:, 1:])
        ratio = torch.exp(logprob - old_logprobs)

        pg_losses = -rewards * ratio
        pg_losses2 = -rewards * torch.clamp(ratio, 1.0 - self.params['cliprange'], 1.0 + self.params['cliprange'])

        loss = torch.mean(torch.max(pg_losses, pg_losses2))
        pg_clipfrac = torch.mean(torch.gt(pg_losses2, pg_losses).double())

        approxkl = .5 * torch.mean((logprob - old_logprobs)**2)
        policykl = torch.mean(logprob - old_logprobs)

        stats = dict(
            loss=dict(total=loss),
            policy=dict(approxkl=approxkl,policykl=policykl, clipfrac=pg_clipfrac, ratio=ratio),
        )
        return loss, flatten_dict(stats)

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
        return stats

    def save_checkpoint(self, directory):
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
