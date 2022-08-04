# GDC
# Copyright 2021-present NAVER Corp.
# Distributed under CC-BY-NC-SA-4.0
__all__ = ['GDCTrainer']


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



class GDCTrainer(BaseTrainer):
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


    def __init__(self, model_cls, tokenizer, sampling_function, features, **params):
        """
        Initialize GDCTrainer.

        Args:
            model (torch.model): pi_theta(x) Policy to be trained e.g. Hugging Face transformer GPT2 model with value head
            orig_model (torch.model): original model before any training: a(x) in the equation above.
                                      e.g. Hugging Face transformer GPT2 original model
            ref_model (torch.model): q(x) a reference modelto calculate off policy DPG
            tokenizer (transformes.Tokenizer): tokenizer to pass to the sampling function
            sampling_function: function that returns samples given a model, tokenizer, scoring function
            features: phi(x) a list of functions that that detect a set of features
            lambdas: lambdas vector, where each lamda corerspond to a feature.

            params (dict or None): DPG parameters for training. Can include following keys:

                'lr' (float): Adam learning rate, default: 1.41e-5
                'batch_size' (int): Number of samples per optimization step, default: 256
                'forward_batch_size' (int): Number of samples forward passed through model at a time. Should be set according to available GPU memory. 
                                            This is used in combination with gradient accumulation to obtain a larger batch size,
                                            default: 16
                'dpg_epochs' (int): Number of optimization epochs per batch of samples, default: 4

        """

        self.params = self.default_params
        self.params.update(params)

        # we pass the sampling method to be able to use it for for the calculaltion of TVD
        # TVD has to be calculated on a new batch than this used to optimize pi_theta
        self.sampling_function = sampling_function
        self.features = features

        self.lambdas = {k:0.0 for k in features} # initialize lambdas with 0
        self.desired_moments = params['desired_moments']

        self.tokenizer = tokenizer


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

        self.optimizer = Adam(self.model.parameters(), lr=self.params['lr'], amsgrad=False)

        # choosing scheduler based on params
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


        ## freezing ref_model, orig_model as we only train pi_theta
        for p1, p2 in zip(self.ref_model.parameters(), self.orig_model.parameters()):
            p1.requires_grad= False
            p2.requires_grad= False

        # will hold all values of P(x) / q(x) for estimating TVD
        self.Z_moving_average = 0
        self.iter = 0
        self.min_kld = float("inf")
        self.min_tvd = float("inf")
        self.is_policy_eval = False

        ### compute gradient accumulation steps
        self.params['gradient_accumulation_steps'] = self.params['batch_size'] // self.params['forward_batch_size']

        #### Compute lambdas
        self.compute_optimal_lambdas(sample_size=self.params["moment_matching_sample_size"])

        ### bootstrap Z
        bstrp_steps = self.params.get('z_bootstrap_steps', 0)

        if bstrp_steps > 0:
            self.bootstrap_z(steps = bstrp_steps)


    def bootstrap_z(self, steps):
        """
        Warmup Z value to accelerate convergence (not used in the paper experiments)
        """

        print ("Boot straping parition function Z... for {} steps".format(steps))

        for i in range(steps):
            _, _, query_tensors, response_tensors, scores = self.sampling_function(self.get_sampling_model(),
            self.tokenizer,
            self.scoring_function,
            self.params['empty_prefix'])

            samples_dict = {'query':[query_tensors], 'response':[response_tensors], 'score':[scores] } # samples dict

            Z_minibatch_mean, _ = self.compute_z(samples_dict) # compute estimate of the parition function Z on minibatches
            self.Z_moving_average = (Z_minibatch_mean + i * self.Z_moving_average) / (i + 1)
            if i % 100 == 0:
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
            scores (torch.tensor): tensor containing the  P(x)/ q(x), shape [batch_size]

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

        for _ in range(self.params['minibatch_epochs']):
            random.shuffle(idxs)

            for i in range(self.params['batch_size'] // fbs):
                fbs_idxs = idxs[i*fbs: (i+1)*fbs]
                epoch_stats = self.train_minibatch(scores[fbs_idxs], query[fbs_idxs],
                                               response[fbs_idxs], model_input[fbs_idxs], step_accumulated=i)

                train_stats.append(epoch_stats)

        train_stats = stack_dicts(train_stats)

        # log current mus (moments)
        #train_stats["scores"] = scores

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

            sample_dict = {'query':[], 'response':[], 'score':[]} # initialize dict

            for _ in range(self.params['q_update_interval']): # sample q_update_interval minibatches from q(x)
                _, _, query_tensors, response_tensors, all_feature_values, scores = self.sampling_function(self.get_sampling_model(),
                    self.tokenizer,
                    self.features,
                    self.lambdas,
                    self.params['prefix'],
                    sample_size=self.params['batch_size'])

                ### partition them to fbs-sized tensors
                for i in range(self.params['batch_size'] // fbs):
                    sample_dict['query'].append(query_tensors[i*fbs: (i+1)*fbs])
                    sample_dict['response'].append(response_tensors[i*fbs: (i+1)*fbs])
                    sample_dict['score'].append(scores[i*fbs: (i+1)*fbs])


            Z_minibatch_mean, Z_minibatch_std = self.compute_z(sample_dict) # compute estimate of the parition function Z on minibatches

            n_previous = self.iter // self.params['q_update_interval'] # of values previously calculated
            self.Z_moving_average = (Z_minibatch_mean + n_previous * self.Z_moving_average) / (n_previous + 1)


            print("Z moving average = ", self.Z_moving_average)

            if self.Z_moving_average > 0.0:
                p_q_tvd = self.compute_tvd_p_q(sample_dict, self.Z_moving_average)
                p_pi_tvd = self.compute_tvd_p_pi(sample_dict, self.Z_moving_average)

                kl_p_q = self.compute_kl(sample_dict, self.Z_moving_average,
                                        pi_theta=False)
                kl_p_pi = self.compute_kl(sample_dict, self.Z_moving_average,
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
        logs['loss/total'] = stats['loss/total'].mean().cpu().numpy()
        logs['lr'] = stats['lr'].mean().cpu().numpy()
        logs['P(x)/q(x)_mean'] = stats['full_reward/total'].mean().cpu().numpy()
        logs['P(x)_mean'] =  stats['P/total'].mean().cpu().numpy()
        logs['q(x)_mean'] = stats['q/total'].mean().cpu().numpy()
        logs['KL(q||a)'] =  stats['kl_q_a/total'].mean().cpu().numpy()
        logs['GPT-2 score'] = -stats['log_a_x/total'].mean().cpu().numpy()
        logs['q_updated?'] = stats['q_updated?']

        ## Computing TVD
        if 'pi_p_tvd' in stats:
            logs['||pi - p||tvd'] = stats['pi_p_tvd']
            logs['||q - p||tvd'] = stats['q_p_tvd']
            logs['KL(p,q)'] = stats['kl_p_q']
            logs['KL(p,pi)'] = stats['kl_p_pi']

            logs['Z_minibatch_mean'] = stats['Z_minibatch_mean']
            logs['Z_moving_average'] = stats['Z_moving_average']
            logs['Z_minibatch_std'] = stats['Z_minibatch_std']

        #logs['b(x~π)_mean'] = torch.mean(scores).cpu().numpy()
        #logs['b(x)_mean'] = torch.mean(stats["scores"]).cpu().numpy()
        #logs['b(x)_std'] = np.std((stats["scores"]).cpu().numpy())

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

            # get log probabilities corresponding to the generated continuation
            tmp = logprobs_from_logits(logits[:,:-1,:], m_input[:,1:])[:, -gen_len:]
            tmp = tmp.detach()
            logprobs.append(tmp)

        return torch.cat(logprobs)

    def train_minibatch(self, scores, query, response, model_input, step_accumulated=0):
        """
        Trains one DPG minibatch
        Args:

            q_logprobs : q(x)
            scores : b(x)

        """
        loss, _ , train_stats  = self.loss(scores, query, response, model_input)
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

    def compute_rewards(self, scores, model_input, gen_len):
        """
        Calculate P(x)/q(x) coefficient

        P(x) = a(x).b(target|x) the energy function
        a(x) = prob. of the sampled sequence by the original model (i.e. gpt-2 orig)
        b(x) = the output of the classifier (scores)

        q(x) = prob. of the sampled sequence by the reference/proposal policy.

        """

        # step1: calculate P(x) = a(x).b(target|x)
        # calculate a(x)

        # move model_input to the same gpu as original model
        model_input = model_input.to(self.params["gpt2_orig_device"])

        orig_logits, _ , _ = self.orig_model(model_input)
        orig_logprob = logprobs_from_logits(orig_logits[:,:-1,:], model_input[:, 1:])

        # @todo: rethink if we should calculate log(a(x)) with or without the query
        orig_logprob = orig_logprob[:, -gen_len:] # (might be not necesary) we only keep prob regardless of the query

        # calculate b(x)
        orig_logprob = orig_logprob.detach() # we don't backprob to original model
        orig_logprob = torch.sum(orig_logprob, dim=-1) # log(a(x)) of shape [batchsize]

        assert scores.shape == orig_logprob.shape

        # we move all variables to the gpu of the policy to be trained "gpt2_device"
        scores = scores.to(self.params["gpt2_ref_device"])
        orig_logprob = orig_logprob.to(self.params["gpt2_ref_device"])


        log_P = orig_logprob.detach() + scores.detach() # Log P(x) = Log a(x)* e^{scores} = Log a(x) + scores

        # step2: calculate q(x)

        model_input = model_input.to(self.params["gpt2_ref_device"])
        ref_logits, _ , _ = self.ref_model(model_input)

        #q_prob = probs_from_logits(ref_logits[:,:-1,:], model_input[:, 1:])
        q_logprobs = logprobs_from_logits(ref_logits[:,:-1,:], model_input[:, 1:])
        q_logprobs = q_logprobs[:, -gen_len:]
        q_logprobs = q_logprobs.detach() # do not backpropagate to q(x)

        q_logprobs = torch.sum(q_logprobs, dim=-1) # Log(q(x)) [Batch size]

        # final reward = exp(Log(P(x)) - Log(q(x)))
        P_over_q = torch.exp(log_P - q_logprobs)

        return P_over_q, log_P, q_logprobs, orig_logprob  # P/q , P(x), q(x), a(x)


    def loss(self, scores, query, response, model_input):
        """
        Calculates DPG loss on a given batch.

        L = (a(x) b(target|x) / q(x)) log pi(x)

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

        ### summing log pi over time steps
        logprob = torch.sum(logprob, dim=-1) # [Batch size]


        #### step2: calculating P(x)/q(x) = (a(x) . b(x|target)) / q(x)
        rewards, log_P, logprob_q, logprob_a = self.compute_rewards(scores, model_input, gen_len)


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


        stats = dict(
            loss= dict(total=loss.detach()),
            full_reward= dict(total=rewards.mean()),
            P = dict(total=torch.exp(log_P).mean()),
            q = dict(total=torch.exp(logprob_q).mean()),
            kl_q_a = dict(total=kl_q_a),
            pi_logprob = dict(total=logprob.detach().mean()),
            log_a_x = dict(total=logprob_a.mean())
            #a_perplexity= dict(total=tor-1 * logprob_a).mean())
        )
        #    policy=dict(entropy=entropy, q_pitheta_kl=policykl),
        #)

        return loss, _, flatten_dict(stats)

    def compute_optimal_lambdas(self, sample_size=4096, n_iters=1000, lr=.5):
        """
        This performs the first step: Constraints --> EBM through self-normalized importance sampling. 
        See section 2.2 in the paper.

        Args:
            sample_size: total number of samples to use for lambda computation
        Returns:
            dicitonary of optimal lambdas per constraint: {'black': lambda_1, 'positive': lambda_2}

        """


        print("Computing Optimal Lambdas for desired moments...")

        bsz = self.params['forward_batch_size']             # replace with fwd batch size from config
        min_nabla_lambda = 0.01
        max_n_iters = n_iters

        feat_names = list(self.features.keys())
        mu_star = self.desired_moments

        mu_star = torch.tensor([mu_star[f] for f in feat_names])
        lambdas = torch.tensor([self.lambdas[f] for f in feat_names])

        # Collect sample_size samples for this:
        list_feature_tensor = []
        list_model_input = []
        for i in  range(sample_size // bsz):
            print("collecting sample: {} / {}".format(i+1, sample_size // bsz))
            _, _, query, response, all_feature_values, exponents = self.sampling_function(self.get_sampling_model(),
                    self.tokenizer,
                    self.features,
                    self.lambdas,
                    self.params['prefix'],
                    sample_size=bsz)

            model_input = torch.cat((query, response), axis=1)
            feature_tensor = torch.stack([all_feature_values[k] for k in feat_names], dim=1) # B x F

            list_model_input.append(model_input)
            list_feature_tensor.append(feature_tensor)

        all_feature_tensor = torch.cat(list_feature_tensor, dim=0)  # [sample_sz x F]

        #### check for zero-occuring features. 
        # If a constraint has not occurred in your sample, no lambdas will be learned for that constraint, so we must check.

        for i, feat in enumerate(feat_names):
            assert all_feature_tensor[:, i].sum().item() > 0, "Feature {feat} hasn't occurred in the samples, use a larger sample size"

        for step in range(max_n_iters):

            # 1. calculate P_over_q batch wise with current lambdas
            ## compute new exponents
            list_P_over_q = []
            for model_input, feature_tensor in zip(list_model_input, list_feature_tensor):
                exponents = lambdas.to(feature_tensor.get_device()).mul(feature_tensor).sum(dim=1) # N  ## compute new exponents
                P_over_q , _, _ ,_ = self.compute_rewards(exponents, model_input, response.shape[1]) # B TODO: use fbs for larger batches
                list_P_over_q.append(P_over_q)

            P_over_q = torch.cat(list_P_over_q, dim=0)

            # 2. compute mu (mean) of features given the current lambda using SNIS
            mu_lambda_numerator = P_over_q.view(1, -1).matmul(all_feature_tensor).squeeze(0) # F
            mu_lambda_denominator = P_over_q.sum()
            mu_lambda = mu_lambda_numerator / mu_lambda_denominator # F

            # 3. Update current Lambdas
            nabla_lambda = mu_star - mu_lambda.cpu()
            err = np.linalg.norm(nabla_lambda.cpu().numpy())
            print("step: %s \t ||nabla_lambda|| = %.6f" %(step, err))
            lambdas = lambdas + lr * nabla_lambda
            print("\tlambdas : {} ".format(self.lambdas))
            print("\tμ: {}".format(mu_lambda))
            print("\tμ*: {}".format(mu_star))

            for i, k in enumerate(feat_names):
                self.lambdas[k] = lambdas[i].item()
            
            ## Check if error is less than tolerance, then break.
            if err < min_nabla_lambda: 
                break


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

