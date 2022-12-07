# GDC
# Copyright 2021-present NAVER Corp.
# Distributed under CC-BY-NC-SA-4.0
__all__ = ['PointwiseGDCTrainer', 'BaselinePointwiseGDCTrainer']


import numpy as np
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD
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
from .diagnostics import GradientMonitor





class PointwiseGDCTrainer(BaseTrainer):
    """
    The DPG trainer uses distributed policy gradient optimization to optimize language models.
    """

    default_params = {
        "lr": 1.41e-5,
        "batch_size": 256,
        "forward_batch_size": 16,
        "minibatch_epochs": 10,
    }

    def get_sampling_model(self):
        return self.ref_model

    def get_policy_model(self):
        return self.model

    def get_eval_model(self):
        return self.ref_model


    def __init__(self, model_cls, tokenizer=None, sampling_function=None, scoring_function=None, **params):
        """
        Initialize PointwiseGDCTrainer.

        Args:
            model (torch.model): pi_theta(x) Policy to be trained e.g. Hugging Face transformer GPT2 model with value head
            orig_model (torch.model): original model before any training: a(x) equation (1) in the paper.
                                      e.g. Hugging Face transformer GPT2 original model
            ref_model (torch.model): q(x) a reference/proposal model to calculate off policy DPG
            tokenizer (transformes.Tokenizer): tokenizer to pass to the sampling function
            sampling_function: function that returns samples given a model, tokenizer, scoring function
            scoring_function: b(x) function that given a text returns a score.

            params (dict or None): DPG parameters for training. Can include following keys:

                'lr' (float): Adam or AdamW learning rate
                'batch_size' (int): Number of samples per optimisation step, default: 256
                'forward_batch_size' (int): Number of samples forward passed through model at a time,
                                            default: 16
                'dpg_epochs' (int): Number of optimisation epochs per batch of samples, default: 4

        """

        super().__init__(tokenizer, sampling_function, scoring_function)

        self.params = self.default_params
        self.params.update(params)

        # double check q_update_criterion
        assert self.params['q_update_criterion'] in ['interval', 'tvd', "kld"]
        # make sure interval is specified
        assert 'q_update_interval' in self.params, "you need to specify an interval to update q"

        # init models
        self.model = model_cls.from_pretrained(self.params['lm_name'], attn_pdrop=self.params['dropout'],
                                            summary_first_dropout=self.params['dropout']).to(self.params['gpt2_device'])

        # original model "a" the one combined with "b" to generate the EBM
        self.orig_model = model_cls.from_pretrained(self.params['lm_name']).to(self.params['gpt2_orig_device'])
        self.orig_model.eval()
        self.ref_model = model_cls.from_pretrained(self.params['lm_name']).to(self.params['gpt2_ref_device'])
        self.ref_model.eval()

        # allow bootstrapping learning rate if "auto" is given
        if self.params['lr'] == "auto":
            self.bootstrap_learning_rate()

        optimizer_cls = eval(params.get('optimizer', 'Adam'))
        self.optimizer = optimizer_cls(self.model.parameters(), lr=self.params['lr'])
        self.is_policy_eval = False

        # adding gradient monitor
        self.gradient_monitor = GradientMonitor(self.model, self.optimizer)

        # choosing scheduler based on params
        scheduler_ = self.params['scheduler']
        assert scheduler_ in ['cosine', 'constant', 'linear', 'cosine_restarts'], "unknown scheduler: {}".format(self.params['scheduler'])
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


        ## freezing ref_model, orig_model
        for p1, p2 in zip(self.ref_model.parameters(), self.orig_model.parameters()):
            p1.requires_grad= False
            p2.requires_grad= False

        # will hold all values of P(x) / q(x) for estimating TVD
        self.Z_moving_average = 0
        self.iter = 0
        self.min_kld = float("inf")
        self.min_tvd = float("inf")

        ### compute gradient accumulation steps
        self.params['gradient_accumulation_steps'] = self.params['batch_size'] // self.params['forward_batch_size']

        ### bootstrap Z
        bstrp_steps = self.params.get('z_bootstrap_steps', 0)

        if bstrp_steps > 0:
            self.bootstrap_z(steps = bstrp_steps)

    def bootstrap_learning_rate(self):
        print("bootstrappin learning rate")
        btstrp_size = 10
        for i in range(btstrp_size):
            self.bootstrap_z(5)
            lr = float(1/self.Z_moving_average * 1e-5 / 666)
            print("step {} of {}: lr = {}".format(i,btstrp_size,lr))
        self.params["lr"] = lr if lr < 5e-5 else 5e-5
        return lr

    def bootstrap_z(self, steps):
        """
        Warmup Z value to accelerate convergence (not used in the paper experiments)
        """
        print ("Boot straping parition function Z... for {} steps".format(steps))

        for i in range(steps):

            sample_dict = self.build_samples_buffer(model = self.get_sampling_model(),
                                                    n_samples = self.params['q_update_interval'] * self.params['batch_size'])
            Z_minibatch_mean, _ = self.compute_z(sample_dict) # compute estimate of the parition function Z on minibatches
            self.Z_moving_average = (Z_minibatch_mean + i * self.Z_moving_average) / (i + 1)
            if i % 5 == 0:
                print("step {}. Z = {}".format(i, self.Z_moving_average))


        print("starting Z values = {}".format(self.Z_moving_average))


    def step(self, query, response, scores):
        """
        This is the main training function. It runs a off-policy DPG (with proposal q) optimization step which includes :

        1. Sample continuations from proposal distribution q(x) which is self.ref_model 
            (already done and output is passed as response)
        2. Compute P(x) / pi(x)
        3. Compute Loss = (P(x) / q(x)) log pi(x) and update policy pi

        Args:
            query (torch.tensor): tensor containing the encoded queries, shape [batch_size, query_length]
            response (torch.tensor): tensor containing the continuation token ids. These were obtained from gdc.gpt2.respond_to_batch().
            scores (torch.tensor): tensor containing the P(x)/ q(x) values, shape [batch_size]

        Returns:
            train_stats (dict): a summary of the training statistics for logging purposes.
        """

        bs = self.params['batch_size']

        gen_len = response.shape[1]
        model_input = torch.cat((query, response), axis=1)

        idxs = list(range(bs))

        # minibatch size
        mbs = bs
        train_stats = []
        fbs = self.params['forward_batch_size']

        grad_monitor = self.params['grad_monitor'] and (self.iter+1) % self.params['grad_monitor_interval'] == 0

        for _ in range(self.params['minibatch_epochs']):
            random.shuffle(idxs)

            for i in range(self.params['batch_size'] // fbs):
                fbs_idxs = idxs[i*fbs: (i+1)*fbs]
                epoch_stats = self.train_minibatch(scores[fbs_idxs], query[fbs_idxs],
                                                    response[fbs_idxs], model_input[fbs_idxs],
                                                    step_accumulated=i, grad_monitor=grad_monitor
                                                    )

                train_stats.append(epoch_stats)


        train_stats = stack_dicts(train_stats)
        train_stats["scores"] = scores

        ##########################################################################################################
        was_q_updated = 0

        ########### Compare pi_theta and q logic and update q #############

        if self.params['q_update_criterion'] == 'interval':

            if (self.iter+1) % self.params['q_update_interval'] == 0:
                print("updating q...")
                self.ref_model.load_state_dict(self.model.state_dict())
                was_q_updated = 1

        elif self.params['q_update_criterion'] in ['tvd', "kld"] and (self.iter+1) % self.params['q_update_interval'] == 0:

            print("sampling {} minibatches for TVD, KLD calculation".format(self.params['q_update_interval']))

            # build samples buffer
            print("epoch {}: comparing pi_theta and q".format(self.iter))

            sample_dict = self.build_samples_buffer(model = self.ref_model, n_samples = self.params['q_update_interval'] * self.params['batch_size'])
            Z_minibatch_mean, Z_minibatch_std = self.compute_z(sample_dict) # compute estimate of the parition function Z on minibatches

            n_previous = self.iter // self.params['q_update_interval'] # of values previously calculated
            self.Z_moving_average = (Z_minibatch_mean + n_previous * self.Z_moving_average) / (n_previous + 1)

            if 'Z_local' in self.params and self.params['Z_local'] == True:
                Z_moving_average = Z_minibatch_mean
                print("Z moving average = ", self.Z_moving_average)
                print("q updates are made locally on Z_mean_batch ")
            else:
                Z_moving_average = self.Z_moving_average
                print("Z moving average = ", self.Z_moving_average)


            if self.Z_moving_average > 0.0:
                p_q_tvd = self.compute_tvd_p_q(sample_dict, Z_moving_average)
                p_pi_tvd = self.compute_tvd_p_pi(sample_dict, Z_moving_average)

                kl_p_q = self.compute_kl(sample_dict, Z_moving_average,
                                        pi_theta=False)
                kl_p_pi = self.compute_kl(sample_dict, Z_moving_average,
                                        pi_theta=True)

                if self.params['q_update_criterion'] == "tvd":
                    if p_q_tvd > p_pi_tvd and (not self.params.get('use_all_previous', False) or p_pi_tvd < self.min_tvd):
                        print("Updating q based on TVD")
                        self.min_tvd = p_pi_tvd
                        self.ref_model.load_state_dict(self.model.state_dict())
                        was_q_updated = 1
                    else:
                        print("Worse TVD, not updating q")
                        was_q_updated = 0

                if self.params['q_update_criterion'] == "kld":
                    if kl_p_q > kl_p_pi and (not self.params.get('use_all_previous', False) or kl_p_pi < self.min_kld):
                        print("Updating q based on KLD")
                        self.min_kld = kl_p_pi
                        self.ref_model.load_state_dict(self.model.state_dict())
                        was_q_updated = 1
                    else:
                        print("Worse KLD, not updating q")
                        was_q_updated = 0


                # if this is the epoch were we calculate the tvd we add tvd info
                train_stats['pi_p_tvd'] = p_pi_tvd
                train_stats['q_p_tvd'] = p_q_tvd
                train_stats["kl_p_q"] = kl_p_q
                train_stats["kl_p_pi"] = kl_p_pi

            train_stats['Z_minibatch_mean'] = Z_minibatch_mean
            train_stats['Z_moving_average'] = self.Z_moving_average
            train_stats['Z_minibatch_std'] = Z_minibatch_std

        train_stats['q_updated?'] = was_q_updated

        self.iter += 1
        return train_stats, self.get_logs_from_stats(train_stats)

    def get_logs_from_stats(self, stats):
        """
        Method to collect stats to log / plot into tracking apps like Neptune or WandB

        Args:
            stats (gdc.core.stack_dicts): all previous statistics

        Returns:
            logs (python dict): a dictinoary holding all required statistics.
        """

        logs = dict()

        logs['loss/total'] = stats['loss/total'].mean().item()
        logs['lr'] = stats['lr'].mean().item()
        logs['P(x)/q(x)_mean'] = stats['P_over_q/total'].mean().item()
        logs['rewards'] = stats['rewards/total'].mean().item()
        logs['P(x)_mean'] = stats['P/total'].mean().item()
        logs['q(x)_mean'] = stats['q/total'].mean().item()
        logs['KL(q||a)'] = stats['kl_q_a/total'].mean().item()
        logs['GPT-2 score'] = -stats['log_a_x/total'].mean().item()
        logs['q_updated?'] = stats['q_updated?']
        num_samples_satisfying_phi = stats['num_samples_satisfying_phi/total'].sum().item()
        logs['num_samples_satisfying_phi/total'] = num_samples_satisfying_phi
        for log_name, value in stats.items():
            if log_name.startswith('grad'):
                logs[log_name] = value.mean()

        ## Computing TVD
        if 'pi_p_tvd' in stats:
            logs['||pi - p||tvd'] = stats['pi_p_tvd']
            logs['||q - p||tvd'] = stats['q_p_tvd']
            logs['KL(p,q)'] = stats['kl_p_q']
            logs['KL(p,pi)'] = stats['kl_p_pi']

            logs['Z_minibatch_mean'] = stats['Z_minibatch_mean']
            logs['Z_moving_average'] = stats['Z_moving_average']
            logs['Z_minibatch_std'] = stats['Z_minibatch_std']

        if 'baseline/total' in stats:
            logs['baseline/total'] = stats['baseline/total'].mean().item()
            # Average advantage of samples for which φ(x) = 1
            logs['advantage_filtered/total'] = (stats['advantage_filtered/total'].sum()/num_samples_satisfying_phi).item()

        if 'advantage/total' in stats:
            logs['advantage/total'] = stats['advantage/total'].mean().item()
            logs['advantage/max'] = stats['advantage/max'].max().item()
            logs['abs_advantage/total'] = stats['abs_advantage/total'].mean().item()
            logs['advantage/std'] = stats['advantage/std'].mean().item()

        #### computing b(x) on Pi theta ############ hack FOR supervised learning exp
        #_, _, _, _ , scores = self.sampling_function(self.model,
        #self.tokenizer,
        #self.scoring_function, self.params['empty_prefix'])

        #logs['b(x~π)_mean'] = torch.mean(scores).cpu().numpy()
        logs['b(x)_mean'] = stats["scores"].mean().item()
        #logs['b(x)_std'] = np.std((stats["scores"]).cpu().numpy())

        ### Gradient Diagnostics
        # get gradient expectations across all batches
        if "mu_grad" in stats and "grad_l22" in stats:
            mugrad =  stats['mu_grad'].mean(axis=0)
            mugrad_l2 = torch.norm(mugrad, p=2)
            logs['mu_grad_l2'] = mugrad_l2
            logs['var(grad)'] = stats['grad_l22'].mean().item() - (mugrad_l2**2).item()

        return logs

    def batched_forward_pass(self, model, model_input, gen_len):
        """
        Calculate a given model outputs in multiple batches.
        given previous input

        Args:
            model (torch.nn.Module): an autoregressive model to use.
            model_input (torch.tensor): shape [batch_size, query_length + gen_length]
            gen_len (int): generation length

        Returns:
            logprobs (torch.tensor): Log probabilities of tokens generated at each continuation step. shape [batch_size, gen_length]. 

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

    def train_minibatch(self, scores, query, response, model_input, step_accumulated=0, grad_monitor=False):
        """
        Train one DPG minibatch
        Args:
            q_logprobs: q(x)
            scores : b(x)

        """
        loss, loss_batch, train_stats  = self.loss(scores, query, response, model_input)

        if grad_monitor:
            grad_info = self.gradient_monitor.compute(loss_batch)
            train_stats.update(grad_info)

        loss.backward()

        if self.params['max_grad_norm']: ### gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params['max_grad_norm'])

        # if last minibatch of the last batch update gradients and update lr
        if (step_accumulated+1) % self.params['gradient_accumulation_steps'] == 0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        train_stats['lr'] = torch.tensor(self.scheduler.get_last_lr()) ## add learnig rate to stats
        return train_stats


    def loss(self, scores, query, response, model_input):
        """
        Calculates DPG loss on a given batch.

        L = (a(x) b(x|positive) / q(x)) log pi(x)

        args:
            q_logprobs (torch.tensor): tensor containing logprobs shape [batch_size, response_length]
            response (torch.tensor): tensor containing response (continuation) token ids, shape [batch_size, response_length]
            scores (torch.tensor): tensor containing b(x_1), b(x_2), .. shape [batch_size]

        returns:
            loss (torch.tensor) []: mean loss value across the batch
            stats (dict) : training statistics
        """

        gen_len = response.shape[1]
        # vpred is the value function of the head
        # in ppo it is used to predict to make the network predict the value function
        # in vanilla DPG its pretty much useless.
        # model_input is query+response

        # movve model input to gpu of the self.model (the trained policy)
        model_input = model_input.to(self.params["gpt2_device"])
        logits, _, vpred = self.model(model_input)

        #### step1: calculating log prob of the policy pi_theta on sampled generations from q theta
        logprob = logprobs_from_logits(logits[:,:-1,:], model_input[:, 1:])
        # only the generation part of the logprobs is needed
        logprob = logprob[:, -gen_len:]  # [batch, response_length]

        ### summing log pi per time step
        logprob = torch.sum(logprob, dim=-1) # [Batch size]


        #### step2: calculating P(x)/q(x) = (a(x) . b(target|x)) / q(x)
        rewards, log_P, logprob_q, logprob_a = self.compute_helper_values(scores, model_input, gen_len)


        assert len(rewards.size()) == 1

        # step3: loss calculation
        rewards = rewards.to(self.params["gpt2_device"])
        rewarded_selected_log_probs = -rewards * logprob
        # todo: replace this with F.nll to account for size_average per sample in the batch
        loss =  torch.mean(rewarded_selected_log_probs) # averaging over all batch samples
        # logging some stats
        stats = dict()

        assert logprob_q.size() == logprob_a.size()
        kl_q_a = torch.mean(logprob_q - logprob_a.to(self.params["gpt2_ref_device"]))
        num_samples_satisfying_phi = scores.sum()

        # Advantage is equal to Reward since no baseline here
        advantage  = rewards

        stats = dict(
            loss= dict(total=loss.detach()),
            num_samples_satisfying_phi=dict(total=num_samples_satisfying_phi),
            P_over_q=dict(total=torch.exp(log_P - logprob_q).mean()),
            rewards=dict(total=rewards.mean()),
            advantage=dict(total=advantage.mean(), std=advantage.std(), max=advantage.max()),
            abs_advantage=dict(total=advantage.abs().mean()),
            P = dict(total=torch.exp(log_P).mean()),
            q = dict(total=torch.exp(logprob_q).mean()),
            kl_q_a = dict(total=kl_q_a),
            pi_logprob = dict(total=logprob.detach().mean()),
            log_a_x = dict(total=logprob_a.mean())
            #a_perplexity= dict(total=tor-1 * logprob_a).mean())
        )
        #    policy=dict(entropy=entropy, q_pitheta_kl=policykl),
        #)

        return loss, rewarded_selected_log_probs, flatten_dict(stats)


    def save_checkpoint(self, directory):
        """
        Saves a checkpoint correctly.

        Args:
            directory (str): path to save ckpt.
        """
        ## creating checkpointting dir if not exists
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        ckpt = super().prepare_checkpoint() # prepare base checkpoint

        ### add additional values to save
        ckpt['ref_model_state'] = self.ref_model.state_dict()
        ckpt['Z_moving_average'] = self.Z_moving_average
        ckpt['min_kld'] = self.min_kld
        ckpt['min_tvd'] = self.min_tvd


        f_path = os.path.join(directory,'checkpoint_last.pt')
        torch.save(ckpt, f_path)

    def load_checkpoint(self, checkpoint_path):
        """
        Loads a saved checkpoint to correctly resume training. 

        Args:
            checkpoint_path (str): path to load from.
        """
        ckpt = torch.load(checkpoint_path)

        self.model.load_state_dict(ckpt['model_state'])
        self.ref_model.load_state_dict(ckpt['ref_model_state'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.iter = ckpt['iter']


        self.Z_moving_average = ckpt['Z_moving_average']
        self.min_kld= ckpt['min_kld']
        self.min_tvd = ckpt['min_kld']


class BaselinePointwiseGDCTrainer(PointwiseGDCTrainer):
    """
    A DPG trainer that subtracts a baseline from the score in its optimisation step. In other words, the loss
    is -[P(x)/q(x) - B] * log_π(x) instead of -P(x)/q(x) * log_π(x).

    Currently, we assume B = Z * π(x)/q(x), where Z is either estimated based on a moving average across epochs or hardcoded.

    """
    def loss(self, scores, query, response, model_input):
        """
        Calculates DPG loss (with baseline) on a given batch.

        args:
            q_logprobs (torch.tensor): tensor containing logprobs shape [batch_size, response_length]
            response (torch.tensor): tensor containing response (continuation) token ids, shape [batch_size, response_length]
            scores (torch.tensor): tensor containing b(x_1), b(x_2), .. shape [batch_size]

        returns:
            loss (torch.tensor) []: mean loss value across the batch
            stats (dict) : training statistics
        """
        gen_len = response.shape[1]
        model_input = model_input.to(self.params["gpt2_device"])
        logits, _, _ = self.model(model_input)

        # step 1: calculating log prob of the policy pi_theta on sampled generations from q theta
        logprob = logprobs_from_logits(logits[:, :-1, :], model_input[:, 1:])

        # only the generation part of the logprobs is needed
        logprob = logprob[:, -gen_len:]  # [batch size, response_length]

        # summing log pi per time step
        logprob = torch.sum(logprob, dim=-1)  # [batch size,]

        # step 2: calculating P(x)/q(x) = (a(x) . b(target|x)) / q(x)
        rewards, log_P, logprob_q, logprob_a = self.compute_helper_values(scores, model_input, gen_len)
        if self.params.get('divide_reward_by_Z', False):
            # Implementing rewards = P(x)/[q(x)Z] instead of P(x)/q(x)
            if self.Z_moving_average == 0:
                self.bootstrap_z(steps=1)
            rewards = torch.exp(log_P - logprob_q - np.log(self.Z_moving_average))
        rewards = rewards.to(self.params["gpt2_device"])
        assert len(rewards.size()) == 1

        # step 3: loss calculation
        baseline = self.get_baseline(logprob.detach(), logprob_q.to(self.params["gpt2_device"]))
        advantage = rewards - baseline
        rewarded_selected_log_probs = -advantage * logprob
        loss = torch.mean(rewarded_selected_log_probs)  # averaging over all batch samples

        # logging some stats
        assert logprob_q.size() == logprob_a.size()
        kl_q_a = torch.mean(logprob_q - logprob_a.to(self.params["gpt2_ref_device"]))
        num_samples_satisfying_phi = scores.sum()
        filtered_advantage = (advantage.cpu() * scores).sum()
        stats = dict(
            loss=dict(total=loss.detach()),
            baseline=dict(total=baseline),
            advantage=dict(total=advantage.mean(), std=advantage.std(), max=advantage.max()),
            abs_advantage=dict(total=advantage.abs().mean()),
            advantage_filtered=dict(total=filtered_advantage),  # Average advantage of samples for which φ(x) = 1
            num_samples_satisfying_phi=dict(total=num_samples_satisfying_phi),
            P_over_q=dict(total=torch.exp(log_P - logprob_q).mean()),
            rewards=dict(total=rewards.mean()),
            P=dict(total=torch.exp(log_P).mean()),
            q=dict(total=torch.exp(logprob_q).mean()),
            kl_q_a=dict(total=kl_q_a),
            pi_logprob=dict(total=logprob.detach().mean()),
            log_a_x=dict(total=logprob_a.mean())
        )
        return loss, rewarded_selected_log_probs, flatten_dict(stats)

    def get_baseline(self, logprob, logprob_q):
        pi_over_q = torch.exp(logprob - logprob_q)
        if self.params.get('Z_in_baseline') is not None:  # hardcode true Z
            Z = self.params['Z_in_baseline']
        else:  # use the moving average estimate of Z
            if self.Z_moving_average == 0:
                self.bootstrap_z(steps=1)
            Z = self.Z_moving_average
        return Z * pi_over_q


