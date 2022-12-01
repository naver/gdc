import random
import os

import numpy as np
from torch.optim import Adam, AdamW, SGD
import torch


from transformers import (get_constant_schedule_with_warmup,
                          get_linear_schedule_with_warmup,
                          get_cosine_schedule_with_warmup,
                          get_cosine_with_hard_restarts_schedule_with_warmup)

from .core import logprobs_from_logits, entropy_from_logits, flatten_dict, stack_dicts
from .base_trainer import BaseTrainer


class PointwiseGDCTrainer(BaseTrainer):
    """
    The DPG trainer uses distributed policy gradient optimization to optimize language models.
    """

    default_params = {
        "lr": 1.41e-5,
        "batch_size": 256,
        "forward_batch_size": 16,
        "minibatch_epochs": 10,
        "baseline": False,
        "online": False,
        "loss_type": 'VPG',
        "cliprange": 0.2,
        "model_kwargs": {}
    }

    def get_sampling_model(self):
        if self.params['online']:
            return self.model
        else:
            return self.ref_model

    def get_policy_model(self):
        return self.model

    def get_eval_model(self):
        if self.params['online']:
            return self.model
        else:
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
        self.model = model_cls.from_pretrained(self.params['lm_name']).to(self.params['gpt2_device'])

        # original model "a" the one combined with "b" to generate the EBM
        self.orig_model = model_cls.from_pretrained(self.params['lm_name']).to(self.params['gpt2_orig_device'])
        self.orig_model.eval()
        if self.params['online']:
            self.ref_model = self.model
        else:
            self.ref_model = model_cls.from_pretrained(self.params['lm_name']).to(self.params['gpt2_ref_device'])
            self.ref_model.eval()

        # allow bootstrapping learning rate if "auto" is given
        if self.params['lr'] == "auto":
            self.bootstrap_learning_rate()

        optimizer_cls = eval(params.get('optimizer', 'Adam'))
        self.optimizer = optimizer_cls(self.model.parameters(), lr=self.params['lr'])
        self.is_policy_eval = False

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

        # freeze ref_model, orig_model
        for p1, p2 in zip(self.ref_model.parameters(), self.orig_model.parameters()):
            if not self.params['online']:
                p1.requires_grad = False
            p2.requires_grad = False

        # will hold all values of P(x) / q(x) for estimating TVD
        self.Z_moving_average = 0
        self.iter = 0
        self.min_kld = float("inf")
        self.min_tvd = float("inf")

        # compute gradient accumulation steps
        self.params['gradient_accumulation_steps'] = self.params['batch_size'] // self.params['forward_batch_size']

        # bootstrap Z
        z_bootstrap_steps = self.params.get('z_bootstrap_steps', 0)

        if z_bootstrap_steps > 0:
            self.bootstrap_z(steps=z_bootstrap_steps)

        if self.params['is_ebm_distributional'] and self.params['Z_c'] is None:
            self.global_lambda = self.estimate_global_lambda_for_old_dpg()

    def estimate_global_lambda_for_old_dpg(self):
        sample_dict = self.build_samples_buffer(model=self.get_eval_model(), n_samples=self.params['batch_size'])
        query_tensors = sample_dict['query']
        response_tensors = sample_dict['response']
        scores = sample_dict['score']
        query_masks = sample_dict['query_mask']
        response_masks = sample_dict['response_mask']
        sampling_model = self.get_sampling_model()
        importance_weights = []
        for idx in range(len(query_tensors)):
            with torch.no_grad():
                logits, _, _ = sampling_model(
                    query=query_tensors[idx],
                    response=response_tensors[idx],
                    query_mask=query_masks[idx]
                )
                log_probs = logprobs_from_logits(
                    logits[:, :-1],
                    response_tensors[idx][:, 1:],
                    mask=response_masks[idx][:, 1:]
                ).sum(dim=1)
                orig_logits, _, _ = self.orig_model(
                    query=query_tensors[idx].to(self.params['gpt2_orig_device']),
                    response=response_tensors[idx].to(self.params['gpt2_orig_device']),
                    query_mask=query_masks[idx].to(self.params['gpt2_orig_device'])
                )
                orig_log_probs = logprobs_from_logits(
                    orig_logits[:, :-1],
                    response_tensors[idx][:, 1:].to(self.params['gpt2_orig_device']),
                    mask=response_masks[idx][:, 1:].to(self.params['gpt2_orig_device'])
                ).sum(dim=1).to(self.params['gpt2_device'])
            importance_weights.append(torch.exp(orig_log_probs - log_probs).detach())
        scores = torch.cat(scores, dim=0)
        importance_weights = torch.cat(importance_weights, dim=0).to(scores.device)
        return self.compute_lambda_using_snis(importance_weights, scores)

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
        print("Boot straping parition function Z... for {} steps".format(steps))

        for i in range(steps):
            sample_dict = self.build_samples_buffer(model = self.get_sampling_model(),
                                                    n_samples = self.params['q_update_interval'] * self.params['batch_size'])
            Z_minibatch_mean, _ = self.compute_z(sample_dict) # compute estimate of the parition function Z on minibatches
            self.Z_moving_average = (Z_minibatch_mean + i * self.Z_moving_average) / (i + 1)
            if i % 5 == 0:
                print("step {}. Z = {}".format(i, self.Z_moving_average))
        print("starting Z values = {}".format(self.Z_moving_average))

    def step(self, query, response, query_mask, response_mask, scores, game_data):
        """
        This is the main training function. It runs a off-policy DPG (with proposal q) optimization step which includes :

        1. Sample continuations from proposal distribution pi(x) or q(x) (if online=False)
        2. Compute P(x) / pi(x)
        3. Compute Loss = (P(x) / q(x)) log pi(x) and update policy pi

        Args:
            query (torch.tensor): tensor containing the encoded queries, shape [batch_size, query_length]
            response (torch.tensor): tensor containing the continuation token ids. These were obtained from cdpg.gpt2.respond_to_batch().
            scores (torch.tensor): tensor containing the P(x)/ q(x) values, shape [batch_size]

        Returns:
            train_stats (dict): a summary of the training statistics for logging purposes.
        """

        bs = self.params['batch_size']
        fbs = self.params['forward_batch_size']
        model_input = torch.cat((query, response), axis=1)
        initial_logprobs = self.batched_forward_pass(
            self.model,
            query.to(self.params['gpt2_device']),
            response.to(self.params['gpt2_device']),
            query_mask.to(self.params['gpt2_device']),
            response_mask.to(self.params['gpt2_device']),
            response.shape[1]
        )
        idxs = list(range(bs))
        train_stats = []
        fbs = self.params['forward_batch_size']

        for _ in range(self.params['minibatch_epochs']):
            random.shuffle(idxs)
            for i in range(self.params['batch_size'] // fbs):
                fbs_idxs = idxs[i*fbs: (i+1)*fbs]
                epoch_stats = self.train_minibatch(
                    initial_logprobs[fbs_idxs],
                    scores[fbs_idxs],
                    query[fbs_idxs],
                    response[fbs_idxs],
                    query_mask[fbs_idxs],
                    response_mask[fbs_idxs],
                    model_input[fbs_idxs],
                    game_data={k: np.array(v)[fbs_idxs] for k, v in game_data.items()},
                    step_accumulated=i,
                )
                train_stats.append(epoch_stats)

        train_stats = stack_dicts(train_stats)
        train_stats["scores"] = scores

        was_q_updated = 0

        # Compare pi_theta and q logic and update q
        if self.params['q_update_criterion'] == 'interval':
            if (self.iter+1) % self.params['q_update_interval'] == 0 and not self.params['online']:
                print("updating q...")
                self.ref_model.load_state_dict(self.model.state_dict())
                was_q_updated = 1

        elif self.params['q_update_criterion'] in ['tvd', "kld"] and (self.iter+1) % self.params['q_update_interval'] == 0:
            print("sampling {} minibatches for TVD, KLD calculation".format(self.params['q_update_interval']))
            # build samples buffer
            print("epoch {}: comparing pi_theta and q".format(self.iter))
            num_samples = self.params.get('num_samples_for_KL') or self.params['q_update_interval'] * self.params['batch_size']
            sample_dict = self.build_samples_buffer(
                model=self.ref_model,
                n_samples=num_samples
            )
            # compute estimate of the parition function Z on minibatches
            Z_minibatch_mean, Z_minibatch_std = self.compute_z(sample_dict, game_data)
            n_previous = self.iter // self.params['q_update_interval']  # of values previously calculated
            self.Z_moving_average = (Z_minibatch_mean + n_previous * self.Z_moving_average) / (n_previous + 1)

            if 'Z_local' in self.params and self.params['Z_local']:
                Z_moving_average = Z_minibatch_mean
                print("Z moving average = ", self.Z_moving_average)
                print("q updates are made locally on Z_mean_batch ")
            else:
                Z_moving_average = self.Z_moving_average
                print("Z moving average = ", self.Z_moving_average)

            if self.params['Z_c'] is not None:
                num_samples_per_query = self.params.get('num_samples_per_query_for_KL')
                num_queries = int(num_samples / num_samples_per_query)
                assert num_samples_per_query >= self.params['forward_batch_size']
                assert num_samples_per_query % self.params['forward_batch_size'] == 0
                num_minibatches_per_query = int(num_samples_per_query/self.params['forward_batch_size'])
                print(f'Computing KL and TVD based on {num_queries} queries with {num_samples_per_query} samples per each')
                p_q_tvds, p_pi_tvds, kl_p_qs, kl_p_pis = [], [], [], []
                for i in range(num_queries):
                    idx = slice(i*num_minibatches_per_query, (i+1)*num_minibatches_per_query)
                    query_sample_dict = {}
                    for key, value in sample_dict.items():
                        query_sample_dict[key] = value[idx]
                    assert np.unique(sample_dict['game_data'][idx.stop - 1]['Z_c']).shape == (
                    1,), 'Z_cs in a minibatch must be all the same!'
                    Z_c_i = sample_dict['game_data'][idx.stop - 1]['Z_c'][0] + 1e-8
                    p_q_tvds.append(self.compute_tvd_p_q(query_sample_dict, Z_c_i))
                    p_pi_tvds.append(self.compute_tvd_p_pi(query_sample_dict, Z_c_i))
                    kl_p_qs.append(self.compute_kl(query_sample_dict, Z_c_i, pi_theta=False))
                    kl_p_pis.append(self.compute_kl(query_sample_dict, Z_c_i, pi_theta=True))
                p_q_tvd = sum(p_q_tvds)/num_queries
                p_pi_tvd = sum(p_pi_tvds)/num_queries
                kl_p_q = sum(kl_p_qs)/num_queries
                kl_p_pi = sum(kl_p_pis)/num_queries
            else:
                print(f'Computing KL and TVD using Z_glob')
                p_q_tvd = self.compute_tvd_p_q(sample_dict, Z_moving_average)
                p_pi_tvd = self.compute_tvd_p_pi(sample_dict, Z_moving_average)
                kl_p_q = self.compute_kl(sample_dict, Z_moving_average, pi_theta=False)
                kl_p_pi = self.compute_kl(sample_dict, Z_moving_average, pi_theta=True)

            if self.params['q_update_criterion'] == "tvd" and not self.params['online']:
                if p_q_tvd > p_pi_tvd and (not self.params.get('use_all_previous', False) or p_pi_tvd < self.min_tvd):
                    print("Updating q based on TVD")
                    self.min_tvd = p_pi_tvd
                    self.ref_model.load_state_dict(self.model.state_dict())
                    was_q_updated = 1
                else:
                    print("Worse TVD, not updating q")
                    was_q_updated = 0

            if self.params['q_update_criterion'] == "kld" and not self.params['online']:
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
            stats (cdpg.core.stack_dicts): all previous statistics

        Returns:
            logs (python dict): a dictinoary holding all required statistics.
        """

        logs = dict()
        logs['loss/total'] = stats['loss/total'].mean().item()
        logs['lr'] = stats['lr'].mean().item()
        logs['P(x)/q(x)_mean'] = stats['P_over_q/total'].mean().item()
        logs['rewards'] = stats['rewards/total'].mean().item()
        logs['advantage'] = stats['advantage/total'].mean().item()
        logs['advantage_min'] = stats['advantage/max'].max().item()
        logs['advantage_max'] = stats['advantage/min'].min().item()
        logs['P(x)_mean'] = stats['P/total'].mean().item()
        logs['q(x)_mean'] = stats['q/total'].mean().item()
        logs['KL(q||a)'] = stats['kl_q_a/total'].mean().item()
        logs['entropy'] = stats['entropy'].mean().item()
        logs['GPT-2 score'] = -stats['log_a_x/total'].mean().item()
        logs['q_updated?'] = stats['q_updated?']
        logs['logpi'] = stats['logpi/total'].mean().item()
        logs['logpi_min'] = stats['logpi/min'].min().item()
        logs['logpi_max'] = stats['logpi/max'].max().item()
        logs['logpi_median'] = stats['logpi/unreduced'].flatten().median().item()
        logs['pi_min'] = stats['pi/min'].min().item()
        logs['pi_max'] = stats['pi/max'].max().item()
        logs['pi_median'] = stats['pi/unreduced'].flatten().median().item()
        num_samples_satisfying_phi = stats['num_samples_satisfying_phi/total'].sum().item()
        logs['num_samples_satisfying_phi/total'] = num_samples_satisfying_phi
        for log_name, value in stats.items():
            if log_name.startswith('grad'):
                logs[log_name] = value.mean()

        # Compute TVD
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

        if self.params['loss_type'] in ['PPO', 'VPG+filtering']:
            logs['entropy'] = stats['entropy'].mean().item()
            logs['policy/clipfrac'] = stats['policy/clipfrac'].mean().item()
            logs['policy/ratio'] = stats['policy/ratio'].mean().item()
            logs['advantage/std'] = stats['advantage/std'].mean().item()

        logs['b(x)_mean'] = stats["scores"].mean().item()
        logs['b(x)_std'] = np.std((stats["scores"]).cpu().numpy())
        return logs

    def batched_forward_pass(self, model, query, response, query_mask, response_mask, gen_len):
        """
        Calculate a given model outputs in multiple batches.
        given previous input

        Args:
            model (torch.nn.Module): an autoregressive model to use.
            query (torch.tensor): shape [batch_size, query_length + gen_length]
            gen_len (int): generation length

        Returns:
            logprobs (torch.tensor): Log probabilities of tokens generated at each continuation step. shape [batch_size, gen_length].

        """

        fbs = self.params['forward_batch_size']
        logprobs = []

        for i in range(int(self.params['batch_size']/fbs)):
            logits, _, _ = model(
                query=query[i*fbs:(i+1)*fbs],
                response=response[i*fbs:(i+1)*fbs],
                query_mask=query_mask[i*fbs:(i+1)*fbs]
            ) # [bsz x len x vocabsize]

            # removing logits corresponding to the query
            tmp = logprobs_from_logits(logits[:, :-1], response[i*fbs:(i+1)*fbs][:, 1:], mask=response_mask[i*fbs:(i+1)*fbs][:, 1:])
            tmp = tmp.detach()
            logprobs.append(tmp)

        return torch.cat(logprobs)

    def train_minibatch(self, old_logprobs, scores, query, response, query_mask, response_mask, model_input, game_data, step_accumulated=0):
        """
        Train one DPG minibatch
        Args:
            q_logprobs: q(x)
            scores : b(x)

        """
        loss, loss_batch, train_stats  = self.loss(old_logprobs, scores, query, response, query_mask, response_mask, model_input, game_data)

        loss.backward()

        if self.params['max_grad_norm']:  # gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params['max_grad_norm'])

        # if last minibatch of the last batch update gradients and update lr
        if (step_accumulated+1) % self.params['gradient_accumulation_steps'] == 0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        train_stats['lr'] = torch.tensor(self.scheduler.get_last_lr())  # add learning rate to stats
        return train_stats

    def loss(self, old_log_pi, scores, query, response, query_mask, response_mask, model_input, game_data):
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

        # move model input to gpu of the self.model (the trained policy)
        query = query.to(self.params["gpt2_device"])
        response = response.to(self.params["gpt2_device"])
        query_mask = query_mask.to(self.params["gpt2_device"])
        response_mask = response_mask.to(self.params["gpt2_device"])
        pi_logits, _, vpred = self.model(
            query=query,
            response=response,
            query_mask=query_mask
        )
        # step1: calculate the log prob of the policy pi_theta on samples from the sampling model
        log_pi = logprobs_from_logits(pi_logits[:, :-1], response[:, 1:], mask=response_mask[:, 1:])
        # step 2: calculate the advantage
        rewards, log_P, logprob_q, logprob_a = self.compute_helper_values(scores, query, response, query_mask, response_mask, game_data)
        advantage = self.compute_advantage(log_P, log_pi, logprob_q, game_data)
        advantage = advantage.to(self.params["gpt2_device"])

        # step 3: calculate the loss
        if self.params['loss_type'] == 'PPO':
            advantage = advantage.unsqueeze(dim=1)
            ratio = torch.exp(log_pi - old_log_pi)
            pg_losses = -advantage * ratio
            pg_losses2 = -advantage * torch.clamp(ratio, 1-self.params['cliprange'], 1+self.params['cliprange'])
            loss = torch.max(pg_losses, pg_losses2).sum(dim=1).mean(dim=0)
            pg_clipfrac = torch.mean(torch.gt(pg_losses2, pg_losses).double())
        elif self.params['loss_type'] == 'VPG':
            # sum log pi across time-steps
            log_pi = log_pi.sum(dim=1)  # [Batch size]
            rewarded_selected_log_probs = -advantage * log_pi
            loss = torch.mean(rewarded_selected_log_probs)  # averaging over all batch samples
        elif self.params['loss_type'] == 'VPG+filtering':
            # sum log pi across time-steps
            log_pi = log_pi.sum(dim=1)  # [Batch size]
            ratio = torch.exp(log_pi - logprob_q.to(self.params['gpt2_device']))
            mask = (ratio > self.params['cliprange']) | (ratio < self.params['cliprange'])
            pg_clipfrac = mask.mean().double()
            rewarded_selected_log_probs = -advantage * log_pi[mask]
            loss = torch.mean(rewarded_selected_log_probs)  # averaging over all batch samples
        else:
            raise TypeError('Incorrect loss type')

        assert logprob_q.size() == logprob_a.size()
        kl_q_a = torch.mean(logprob_q - logprob_a.to(logprob_q.device))
        num_samples_satisfying_phi = scores.sum()
        if self.params['online']:
            rewards = torch.exp(log_P.to(self.params["gpt2_device"]) - log_pi.detach())
        else:
            rewards = rewards.to(self.params["gpt2_device"])
        stats = dict(
            loss=dict(total=loss.detach()),
            advantage=dict(total=advantage.detach(), max=advantage.detach().max(), min=advantage.detach().min()),
            num_samples_satisfying_phi=dict(total=num_samples_satisfying_phi),
            rewards=dict(total=rewards.mean()),
            P_over_q=dict(total=torch.exp(log_P - logprob_q.to(log_P.device)).mean()),
            P=dict(total=torch.exp(log_P.double()).mean()),
            q=dict(total=torch.exp(logprob_q.double()).mean()),
            kl_q_a=dict(total=kl_q_a),
            pi_logprob=dict(total=log_pi.detach().mean()),
            log_a_x=dict(total=logprob_a.mean()),
            entropy=entropy_from_logits(pi_logits.detach()).mean(),
            logpi=dict(
                total=log_pi.double().mean(),
                max=log_pi.double().max(),
                min=log_pi.double().min(),
                unreduced=log_pi.double()
            ),
            pi=dict(
                total=log_pi.double().exp().mean(),
                max=log_pi.double().exp().max(),
                min=log_pi.double().exp().min(),
                unreduced=log_pi.double().exp()
            )
        )
        if self.params['loss_type'] in ['PPO', 'VPG+filtering']:
            stats['policy'] = dict(clipfrac=pg_clipfrac, ratio=ratio)

        return loss, rewarded_selected_log_probs, flatten_dict(stats)

    def compute_advantage(self, log_P, log_pi, logprob_q, game_data):
        if self.params['online']:
            log_pseudoreward = log_P.to(self.params["gpt2_device"]) - log_pi.sum(dim=1)
        else:
            log_pseudoreward = (log_P - logprob_q).to(self.params["gpt2_device"])

        if self.params['baseline']:
            baseline = self.get_baseline(log_pi.detach(), logprob_q.to(self.params["gpt2_device"]))
            return torch.exp(log_pseudoreward) - baseline
        else:
            if self.params.get('Z_c') is not None:
                Z_c = torch.FloatTensor(game_data['Z_c']).to(self.params["gpt2_device"]) + 1e-4
                return torch.exp(log_pseudoreward - Z_c.log())
            else:
                return torch.exp(log_pseudoreward)

    def get_baseline(self, logprob, logprob_q):
        if self.params.get('Z_c') is None:
            if self.Z_moving_average == 0:
                self.bootstrap_z(steps=1)
            Z_glob = self.Z_moving_average
        if self.params['online']:
            if self.params.get('Z_c') is not None:
                return 1
            else:
                return torch.tensor([Z_glob]).to(self.params["gpt2_device"])
        else:
            pi_over_q = torch.exp(logprob.sum(dim=1) - logprob_q)
            if self.params.get('Z_c') is not None:
                return pi_over_q
            else:
                return Z_glob * pi_over_q

    def save_checkpoint(self, directory):
        """
        Saves a checkpoint correctly.

        Args:
            directory (str): path to save ckpt.
        """
        if not os.path.exists(directory):  # create a checkpointing dir if not exists
            os.makedirs(directory, exist_ok=True)
        ckpt = super().prepare_checkpoint()  # prepare base checkpoint

        # add additional values to save
        ckpt['ref_model_state'] = self.ref_model.state_dict()
        ckpt['Z_moving_average'] = self.Z_moving_average
        ckpt['min_kld'] = self.min_kld
        ckpt['min_tvd'] = self.min_tvd

        f_path = os.path.join(directory, 'checkpoint_last.pt')
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
        self.min_kld = ckpt['min_kld']
        self.min_tvd = ckpt['min_kld']
