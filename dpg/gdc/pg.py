# GDC
# Copyright 2021-present NAVER Corp.
# Distributed under CC-BY-NC-SA-4.0

__all__ = ['PGTrainer']


import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
import torch
import collections
import time, os
import random
import copy

from .core import (logprobs_from_logits,
                      probs_from_logits,
                         whiten,
                         clip_by_value,
                         entropy_from_logits,
                         flatten_dict,
                         average_torch_dicts,
                         stats_to_np,
                         stack_dicts,
                         add_suffix)

from .base_trainer import BaseTrainer
from transformers import (get_constant_schedule_with_warmup,
                          get_linear_schedule_with_warmup,
                          get_cosine_schedule_with_warmup,
                          get_cosine_with_hard_restarts_schedule_with_warmup)







class PGTrainer(BaseTrainer):
    """
    The PGTrainer uses policy gradient with Monte Calro rollout to compute the rewards optimization to optimise language models.
    """

    default_params = {
        "lr": 1.41e-5,
        "batch_size": 256,
        "forward_batch_size": 16,
        "minibatch_epochs": 10,
    }

    def get_policy_model(self):
        return self.model

    def get_sampling_model(self):
        return self.model

    def get_eval_model(self):
        return self.model

    def __init__(self, model_cls, tokenizer, sampling_function, scoring_function, **params):
        """
        Initialize PGTrainer.

        Args:
            model (torch.model): pi_theta(x) Policy to be trained e.g. Hugging Face transformer GPT2 model with value head
            orig_model (torch.model): original model before any training: a(x) in the equation above. e.g. Hugging Face transformer GPT2 original model

            params (dict or None): Vanilla PG parameters for training. Can include following keys:

                'lr' (float): Adam learning rate, default: 1.41e-5
                'batch_size' (int): Number of samples per optimisation step, default: 256
                'forward_batch_size' (int): Number of samples forward passed through model at a time, default: 16
                'minibatch_epochs' (int): Number of optimisation epochs per batch of samples, default: 4

        """

        super().__init__(tokenizer=tokenizer, sampling_function=sampling_function, scoring_function=scoring_function)

        self.params = self.default_params
        self.params.update(params)

        # pi_theta policy to be learned
        self.model = model_cls.from_pretrained(params['lm_name']).to(params['gpt2_device'])

        # original model for computing kl(pi||a)
        self.orig_model = model_cls.from_pretrained(params['lm_name']).to(params['gpt2_orig_device'])

        self.ref_model = self.orig_model
        self.is_policy_eval = True


        #optimzier

        self.optimizer = Adam(self.model.parameters(), lr=self.params['lr'])

        # scheduler
        scheduler_ = self.params['scheduler']
        assert scheduler_ in ['cosine', 'constant', 'linear'], "unknown scheduler: {}".format(self.params['scheduler'])

        if scheduler_ == 'constant':
            self.scheduler = get_constant_schedule_with_warmup(self.optimizer, self.params['warmup_steps'])
        elif scheduler_ == 'cosine':
            print("Cosine scheduler...")
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, self.params['warmup_steps'],
                                                            self.params['steps']//self.params['batch_size'])
        elif scheduler_ == 'linear':
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, self.params['warmup_steps'])

        self.params['gradient_accumulation_steps'] = self.params['batch_size'] // self.params['forward_batch_size']


    def step(self, query, response, scores):
        """
        Run a PG optimisation step which includes :

        1. Sample continuations from π(x) which is self.model (already done and output is passed as response)
        3. Loss = b(x) log pi(x)

        args:
            query (torch.tensor): tensor containing the encoded queries, shape [batch_size, query_length]
            response (torch.tensor): tensor containing the encoded responses we get this from respond to
            batch function in gdc.gpt2.respond_to_batch
            scores (torch.tensor): tensor containing the (Rewards), shape [batch_size]

        returns:
            train_stats (dict): a summary of the training statistics
        """

        bs = self.params['batch_size']
        gen_len = response.shape[1]
        model_input = torch.cat((query, response), axis=1)

        idxs = list(range(bs))
        train_stats = []

        #minibatch size
        fbs = self.params['forward_batch_size']

        for _ in range(self.params['minibatch_epochs']):
            random.shuffle(idxs)

            for i in range(self.params['batch_size'] // fbs):
                fbs_idxs = idxs[i*fbs: (i+1)*fbs]
                epoch_stats = self.train_minibatch(scores[fbs_idxs], query[fbs_idxs],
                                               response[fbs_idxs], model_input[fbs_idxs])
                train_stats.append(epoch_stats)

        train_stats = stack_dicts(train_stats)
        train_stats["scores"] = scores

        sample_dict = self.build_samples_buffer(model= self.ref_model, n_samples = self.params['batch_size'])
        Z_minibatch_mean, Z_minibatch_std = self.compute_z(sample_dict) # compute estimate of the parition function Z on minibatches

        n_previous = self.iter
        self.Z_moving_average = (Z_minibatch_mean + n_previous * self.Z_moving_average) / (n_previous + 1)

        # increment counter
        self.iter += 1
        return train_stats, self.get_logs_from_stats(train_stats)


    def get_logs_from_stats(self, stats):
        """Method to collect stats to log / plot into tracking apps like
            Neptune or WandB
            args:
            all_stats (gdc.core.stack_dicts): all previous statistics
        """
        logs = dict()
        logs['loss'] = stats['loss/total'].mean().cpu().numpy()
        logs['b(x)_mean'] = torch.mean(stats["scores"]).cpu().numpy()
        logs['b(x)_std'] = torch.std(stats["scores"]).cpu().numpy()
        logs['KL(π||a)'] =  stats['kl_pi_a/total'].mean().cpu().numpy()
        logs['GPT-2 score'] = -stats['log_a_x/total'].mean().cpu().numpy()
        logs['reward'] = stats['reward/total'].mean().item()

        logs['lr'] = stats['lr'].mean().cpu().numpy()


        return logs

    def batched_forward_pass(self, model, model_input, gen_len):
        """Calculate a given model outputs in multiple batches.
        given previous input

        args:
            model_input (torch.tensor): shape [batch_size, query_length + gen_length] output of model,
            gen_len (int): generation length
        """

        bs = self.params['batch_size']
        fbs = self.params['forward_batch_size']
        logprobs = []

        for i in range(int(self.params['batch_size']/fbs)):
            m_input = model_input[i*fbs:(i+1)*fbs]
            # feed the tokens to the model and get the output logits
            logits, _, _ = model(m_input) # [bsz x len x vocabsize]

            # removing logits corresponding to the query
            tmp = logprobs_from_logits(logits[:,:-1,:], m_input[:,1:])[:, -gen_len:]
            tmp = tmp.detach()
            logprobs.append(tmp)

        return torch.cat(logprobs)

    def train_minibatch(self, scores, query, response, model_input):
        """
        Train on one minibatch
        Args:
            q_logprobs: q(x)
            scores : b(x)

        """
        loss, _ , train_stats  = self.loss(scores, query, response, model_input)
        loss.backward()

        if (self.iter+1) % self.params['gradient_accumulation_steps'] == 0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()


        train_stats['lr'] = torch.tensor(self.scheduler.get_last_lr()) ## add learnig rate to stats

        return train_stats


    def loss(self, scores, query, response, model_input):
        """
        Calculate dpg loss

        (a(x) b(x|positive) / q(x)) log pi(x)

        args:
            response (torch.tensor): tensor containing the encoded responses,
            shape [batch_size, response_length]
            rewards (torch.tensor): tensor containing the (Rewards), shape [batch_size]

        returns:
            loss (torch.tensor) []: mean loss value across the batch
            stats (dict) : training statistics
        """

        gen_len = response.shape[1]

        # model_input is query+response

        ### original probability a(x)
        model_input = model_input.to(self.params["gpt2_orig_device"])

        orig_logits, _ , _ = self.orig_model(model_input)
        orig_logprob = logprobs_from_logits(orig_logits[:,:-1,:], model_input[:, 1:])
        # @todo: rethink if we should calculate log(a(x)) with or without the query
        orig_logprob = orig_logprob[:, -gen_len:] # (might be not necesary) we only keep prob regardless of the query
        # calculate b(x)
        orig_logprob = orig_logprob.detach() # we don't backprob to original model
        orig_logprob = torch.sum(orig_logprob, dim=-1) # log(a(x)) of shape [batchsize]

        # movve model input to gpu of the self.model (the trained policy)
        model_input = model_input.to(self.params["gpt2_device"])
        logits, _, vpred = self.model(model_input)

        #### step1: calculating log prob of the policy pi_theta on sampled generations from q theta
        logprob = logprobs_from_logits(logits[:,:-1,:], model_input[:, 1:])
        # only the generation part of the logprobs is needed
        logprob = logprob[:, -gen_len:]  # [batch, response_length]

        ### summing log pi per time step
        logprob = torch.sum(logprob, dim=-1) # [Batch size]

        if self.params.get('use_P_as_reward', False):
            _, log_P, logprob_q, logprob_a = self.compute_helper_values(scores, model_input, gen_len)

            rewards = torch.exp(log_P) * 1e25
            # replace zeros in scores with -1000000
            #scores[scores ==0] = 1e-6
            #rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-10) # normalize rewards
            #rewards = logprob_a * scores.to(self.params['gpt2_orig_device'])

        else:
            rewards = scores # R(x) = b(x)


        assert len(rewards.size()) == 1

        # step3: loss calculation
        rewards = rewards.to(self.params["gpt2_device"])
        rewarded_selected_log_probs = -rewards * logprob
        # todo: replace this with F.nll to account for size_average per sample in the batch
        loss =  torch.mean(rewarded_selected_log_probs) # averaging over all batch samples

        # logging some stats
        kl_pi_a = torch.mean(logprob - orig_logprob.to(self.params["gpt2_device"])).detach()
        stats = dict(
            loss= dict(total=loss.detach()),
            full_reward= dict(total=rewards.mean()),
            kl_pi_a=dict(total=kl_pi_a),
            log_a_x=dict(total=orig_logprob.mean()),
            reward = dict(total=rewards.mean())
            )

        return loss, _, flatten_dict(stats)



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
