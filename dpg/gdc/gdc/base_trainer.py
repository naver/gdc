# GDC
# Copyright 2021-present NAVER Corp.
# Distributed under CC-BY-NC-SA-4.0
__all__ = ['BaseTrainer']


import numpy as np
import torch

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
import abc



class BaseTrainer:
    """
    The BaseTrainer Class
    """

    def __init__(self, tokenizer=None, sampling_function=None, scoring_function=None, **params):
        """
        tokenizer (transformes.Tokenizer): tokenizer to be used inside trainer
        sampling_function (python func): a function used to sample continuations from a given autoregressive model
        scoring_function (python func): a function to score a collection of samples. Obtained from Scorer.get_scoring_fn()
        params: additional keyword arguments
        """
        self.tokenizer= tokenizer
        self.sampling_function = sampling_function
        self.scoring_function = scoring_function


        self.sample_buffer = None
        self.Z_moving_average = 0.0
        self.iter=0



    def build_samples_buffer(self, model, n_samples):
        """
        Samples n_samples from given model, and returns the sample buffer.
        """

        fbs = self.params['forward_batch_size']
        n_batches = n_samples // self.params['batch_size']

        samples_dict = {'query':[], 'response':[], 'score':[]}

        pointwise_only = not hasattr(self, 'lambdas') ## check if we are in the purely pointwise case or not
        for _ in range(n_batches): # sample q_update_interval minibatches from q(x)

            if pointwise_only:
                _, _, query_tensors, response_tensors, scores = self.sampling_function(model,
                                                                                self.tokenizer,
                                                                                self.scoring_function,
                                                                                self.params['empty_prefix'], top_p=1.0)
            else:
                _, _, query_tensors, response_tensors, \
                    _, scores = self.sampling_function(self.get_sampling_model(),
                                                tokenizer=self.tokenizer,
                                                features=self.features,
                                                lambdas=self.lambdas,
                                                prefix_str=self.params['prefix'],
                                                sample_size=self.params['batch_size'])

            ### partition them to fbs-sized tensors
            for i in range(self.params['batch_size'] // fbs):
                samples_dict['query'].append(query_tensors[i*fbs: (i+1)*fbs])
                samples_dict['response'].append(response_tensors[i*fbs: (i+1)*fbs])
                samples_dict['score'].append(scores[i*fbs: (i+1)*fbs])

        self.sample_buffer = samples_dict
        return samples_dict

    def compute_z(self, sample_dict):
        """
        Computes Z (partition function) estimate through Z = E x~q(x) [P(x) / q(x)]
        Args:
          sample_dict (python dict): dictionary of samples from self.ref_model
        Returns:
            Z_estimate : average of Z estimate per samples in self.response_archive
            Z_std: standard deviation of Z estimate across the samples in self.reponse_archive

        """

        query_mbs, response_mbs, scores_mbs = sample_dict['query'], sample_dict['response'], sample_dict['score']

        P_q_s = []
        values = []

        ### iterate over stored queries, responses, and scores
        for query, response, scores in zip(query_mbs, response_mbs, scores_mbs):
            model_input = torch.cat((query, response), axis=1)
            gen_len = response.shape[1]

            P_over_q, _, _, _ = self.compute_helper_values(scores, model_input, gen_len)
            P_over_q = P_over_q.cpu().numpy()
            values.extend(P_over_q.tolist())
            assert not np.isnan(P_over_q).any()

        return np.mean(values), np.std(values)


    def compute_kl(self, sample_dict, Z_estimate, pi_theta=False):
        """
        estimates KL divergence
        
        Args:
          sample_dict (python dict): sampels from ref_model
          Z_estimate (float): estimation of the partition function Z
          pi_theta (boolean): Flag to determine whether to calculate kld(p,q) if pi_theta is False else kld(p,pi_theta)
          
        Returns:
          kld (float): estimated KL divergence
        """

        # get queries / responses / scores for mini batch
        query_mbs, response_mbs, scores_mbs = sample_dict['query'], \
                                                sample_dict['response'], \
                                                sample_dict['score']
        values = []

        for query, response, scores in zip(query_mbs, response_mbs, scores_mbs):
            model_input = torch.cat((query, response), axis=1)
            gen_len = response.shape[1]

            ## computing q(x) and P(x)/q(x)
            P_over_q, log_P, q_logprobs, _ = self.compute_helper_values(scores, model_input, gen_len)
            log_q_x = q_logprobs.cpu().numpy().tolist()
            P_over_q = P_over_q.cpu().numpy().tolist()
            log_P = log_P.detach().cpu().numpy().tolist()

            # if pi_theta is none then calculate kl(q,p)
            # else calculate kl(pi_theta,p)
            if pi_theta is False:
                log_pi_x = log_q_x
            else:
                ## Computing π(x)
                model_input = model_input.to(self.params["gpt2_device"])
                logits, _, _ = self.model(model_input)
                # step1: calculating log prob of the policy pi_theta on
                # sampled generations from q theta
                pi_logprob = logprobs_from_logits(logits[:,:-1,:],
                                                    model_input[:, 1:])
                # only the generation part of the logprobs is needed
                pi_logprob = pi_logprob[:, -gen_len:]  # [batch, response_length]
                ### summing log pi per time step
                pi_logprob = torch.sum(pi_logprob, dim=-1) # [Batch size]
                log_pi_x = pi_logprob.detach().cpu().numpy().tolist()

            for i, (lg_P, lg_pi, P_q, lg_q) in enumerate(zip(log_P, log_pi_x,
                                                        P_over_q, log_q_x)):
                # calculate the lhs part of the kl equation
                # if P  = 0 , P_q = 0 and lg_p = -inf  the whole lhs = 0 x log(0) = 0
                if P_q == 0 and lg_P == -float("inf"):
                    kl_lhs = 0.0
                else:
                    kl_lhs = P_q * (lg_P - lg_pi)
                values.append(kl_lhs)

        kld = -1 * np.log(Z_estimate) + (1/Z_estimate) * np.mean(values)

        return kld


    def compute_tvd_p_q(self, sample_dict, Z_estimate, pi_theta=False):
        """
        Estimates Total Variation Distance (TVD) by  ||p - q||tvd = 0.5* E x~q(x) |1- P(x)/(Z q(x))|
        
        Args:
          sample_dict (python dict): sampels from ref_model
          Z_estimate (float): estimation of the partition function Z
          pi_theta (boolean): Flag to determine whether to calculate kld(p,q) if pi_theta is False else kld(p,pi_theta)
          
        Returns:
          TVD (float): estimated TVD
        """
        query_mbs, response_mbs, scores_mbs = sample_dict['query'], sample_dict['response'], sample_dict['score']
        values = []

        ### iterate over stored queries, responses, and scores
        for query, response, scores in zip(query_mbs, response_mbs, scores_mbs):
            model_input = torch.cat((query, response), axis=1)
            gen_len = response.shape[1]

            P_over_q, _, _, _ = self.compute_helper_values(scores, model_input, gen_len)

            # average over minibatch
            #P_over_q = P_over_q.mean().item()

            P_over_q = P_over_q.detach().cpu().numpy().tolist()

            for P_q in P_over_q:
                values.append(np.abs(1 - P_q / Z_estimate))

        return 0.5 * np.mean(values)


    def compute_tvd_p_pi(self, sample_dict, Z_estimate):
        """
        Computes ||p - π||tvd = 0.5* E x~q(x) |π(x)/q(x) - P(x)/Zq(x)|
        """
        query_mbs, response_mbs, scores_mbs = sample_dict['query'], sample_dict['response'], sample_dict['score']
        values = []

        for query, response, scores in zip(query_mbs, response_mbs, scores_mbs):
            model_input = torch.cat((query, response), axis=1)
            gen_len = response.shape[1]

            ## Computing π(x)
            model_input = model_input.to(self.params["gpt2_device"])
            logits, _, _ = self.model(model_input)

            #### step1: calculating log prob of the policy pi_theta on sampled generations from q theta
            pi_logprob = logprobs_from_logits(logits[:,:-1,:], model_input[:, 1:])
            # only the generation part of the logprobs is needed
            pi_logprob = pi_logprob[:, -gen_len:]  # [batch, response_length]

            ### summing log pi per time step
            pi_logprob = torch.sum(pi_logprob, dim=-1) # [Batch size]
            log_pi_x = pi_logprob.detach().cpu().numpy().tolist()


            P_over_q, _, q_logprobs, _ = self.compute_helper_values(scores, model_input, gen_len)
            log_q_x = q_logprobs.detach().cpu().numpy().tolist()
            P_over_q = P_over_q.detach().cpu().numpy().tolist()

            for i, (lg_pi, P_q, lg_q) in enumerate(zip(log_pi_x, P_over_q, log_q_x)):

                v = np.abs(np.exp(lg_pi - lg_q) - P_q / Z_estimate)
                values.append(np.abs(np.exp(lg_pi - lg_q) - P_q / Z_estimate))

        return 0.5 * np.mean(values)

    def compute_helper_values(self, scores, model_input, gen_len):
        """
        Calculate several useful quantities
        
        Args:
          scores (torch.tensor): a tensor of scores i.e b(x)
          model_input (torch.tensor): torch tensor of token ids  (query + continuation)
          gen_len (int): length of generated continuations
          
        Returns: 
          P_over_q (torch.tensor): tensor of P(x)/q(x) values, where P(x) = a(x).b(target|x) the energy function
          log_P (torch.tensor): tensor of Log P(x) values
          q_logprobs (torch.tensor): tensor of Log q(x) values, where q(x) is prob. of the sampled sequence by the reference policy
          orig_logprob (torch.tensor): tensor of Log a(x) values, a(x) is prob. of the sampled sequence by the original model (i.e. gpt-2 orig)    

        """

        # step1: calculate P(x) = a(x).b(target|x)
        # calculate a(x)

        # move model_input to the same gpu as original model
        model_input = model_input.to(next(self.orig_model.parameters()).device)

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

        if hasattr(self, 'lambdas'):
            # not in the purely poitnwise_only case 
            scores_ = scores.detach()
        else:
            scores_ = torch.log(scores.detach())

        log_P = orig_logprob.detach() + scores_ # Log P(x) = Log a(x)* scores = Log a(x) + log(scores)

        # step2: calculate q(x)

        model_input = model_input.to(next(self.ref_model.parameters()).device)
        ref_logits, _ , _ = self.ref_model(model_input)

        #q_prob = probs_from_logits(ref_logits[:,:-1,:], model_input[:, 1:])
        q_logprobs = logprobs_from_logits(ref_logits[:,:-1,:], model_input[:, 1:])
        q_logprobs = q_logprobs[:, -gen_len:]
        q_logprobs = q_logprobs.detach() # do not backpropagate to q(x)

        q_logprobs = torch.sum(q_logprobs, dim=-1) # Log(q(x)) [Batch size]

        # final reward = exp(Log(P(x)) - Log(q(x)))
        P_over_q = torch.exp(log_P - q_logprobs)

        return P_over_q, log_P, q_logprobs, orig_logprob  # R(x) ,P(x), q(x), a(x)


    def prepare_checkpoint(self):

        ckpt = {
                'iter': self.iter,
                'model_state': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
            }

        return ckpt

    def eval_kl_a(self):

        """
        Computes and returns KL divergence with the original unmodified language model (i.e GPT2)
        """
        sample_dict = self.build_samples_buffer(model = self.get_eval_model(), n_samples = self.params['batch_size'])

        query_mbs, response_mbs, scores_mbs = sample_dict['query'], sample_dict['response'], sample_dict['score']
        values = []

        eval_model = self.get_eval_model()
        eval_model_device = next(eval_model.parameters()).device

        for query, response, scores in zip(query_mbs, response_mbs, scores_mbs):
            model_input = torch.cat((query, response), axis=1)
            gen_len = response.shape[1]

                ## Computing eval_model(x)
            model_input = model_input.to(eval_model_device)
            logits, _, _ = eval_model(model_input)

            #### step1: calculating log prob of the policy pi_theta on sampled generations from q theta
            model_logprob = logprobs_from_logits(logits[:,:-1,:], model_input[:, 1:])
            # only the generation part of the logprobs is needed
            model_logprob = model_logprob[:, -gen_len:]  # [batch, response_length]

            ### summing log pi per time step
            model_logprob = torch.sum(model_logprob, dim=-1) # [Batch size]
            log_model_x = model_logprob

            _, _, _, orig_logprob = self.compute_helper_values(scores, model_input, gen_len)
            orig_logprob= orig_logprob

            values.append(torch.mean(model_logprob - orig_logprob.to(eval_model_device)).item())

        return np.mean(values)


    def eval_kl_p(self):

        """
        Compute KL(P || .) on a batch of samples
        samples come from reference model
        """
        # compute KL(P || pi)
        sample_dict = self.build_samples_buffer(model = self.ref_model, n_samples = self.params['batch_size'])
        return self.compute_kl(sample_dict, self.Z_moving_average, pi_theta=self.is_policy_eval)



    @abc.abstractmethod
    def save_checkpoint(self):
        pass


    @abc.abstractmethod
    def load_checkpoint(self):
        pass








