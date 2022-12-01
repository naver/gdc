import abc

import numpy as np
import torch

from .core import logprobs_from_logits


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
        self.iter = 0

    def build_samples_buffer(self, model, n_samples, force_condition=None):
        """
        Samples n_samples from given model, and returns the sample buffer.
        """

        fbs = self.params['forward_batch_size']
        n_batches = n_samples // self.params['batch_size']

        samples_dict = {'query':[], 'response':[], 'score':[], 'query_mask': [], 'response_mask': [], 'game_data': []}

        for _ in range(n_batches): # sample q_update_interval minibatches from q(x)
            game_data, _, query_tensors, response_tensors, query_masks, response_masks, scores = self.sampling_function(
                model,
                self.tokenizer,
                self.scoring_function,
                self.params['empty_prefix'], top_p=1.0,
                params=self.params,
                force_condition=force_condition,
                num_samples_per_query=self.params.get('num_samples_per_query_for_KL', 1),
                decoding_params=self.params.get('train_decoding_params')
            )
            self.compute_Z_c_using_MC(
                query_tensors,
                response_tensors,
                query_masks,
                response_masks,
                game_data,
                scores,
                num_samples_per_query=self.params.get('num_samples_per_query_for_KL', 1)
            )

            # partition them to fbs-sized tensors
            for i in range(self.params['batch_size'] // fbs):
                samples_dict['query'].append(query_tensors[i*fbs: (i+1)*fbs])
                samples_dict['response'].append(response_tensors[i*fbs: (i+1)*fbs])
                samples_dict['query_mask'].append(query_masks[i*fbs: (i+1)*fbs])
                samples_dict['response_mask'].append(response_masks[i*fbs: (i+1)*fbs])
                samples_dict['score'].append(scores[i*fbs: (i+1)*fbs])
                samples_dict['game_data'].append(
                    {k: np.array(v)[i * fbs: (i + 1) * fbs]
                     for k, v in game_data.items()}
                )
        self.sample_buffer = samples_dict
        return samples_dict

    def compute_z(self, sample_dict, game_data):
        """
        Computes Z (partition function) estimate through Z = E x~q(x) [P(x) / q(x)]
        Args:
          sample_dict (python dict): dictionary of samples from self.ref_model
        Returns:
            Z_estimate : average of Z estimate per samples in self.response_archive
            Z_std: standard deviation of Z estimate across the samples in self.reponse_archive

        """

        query_mbs = sample_dict['query']
        response_mbs = sample_dict['response']
        scores_mbs = sample_dict['score']
        query_mask_mbs = sample_dict['query_mask']
        response_mask_mbs = sample_dict['response_mask']
        game_data_mbs = sample_dict['game_data']
        values = []
        # iterate over stored queries, responses, and scores
        for query, response, query_mask, response_mask, scores, game_data in zip(query_mbs, response_mbs, query_mask_mbs, response_mask_mbs, scores_mbs, game_data_mbs):
            P_over_q, _, _, _ = self.compute_helper_values(scores, query, response, query_mask, response_mask, game_data)
            P_over_q = P_over_q.cpu().numpy()
            values.extend(P_over_q.tolist())
            assert not np.isnan(P_over_q).any()
        return np.mean(values), np.std(values)

    def compute_kl(self, sample_dict, Z_estimate, pi_theta=False, cross_entropy=False):
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
        query_mbs = sample_dict['query']
        response_mbs = sample_dict['response']
        scores_mbs = sample_dict['score']
        query_mask_mbs = sample_dict['query_mask']
        response_mask_mbs = sample_dict['response_mask']
        game_data_mbs = sample_dict['game_data']
        values = []

        for query, response, query_mask, response_mask, scores, game_data in zip(query_mbs, response_mbs, query_mask_mbs, response_mask_mbs, scores_mbs, game_data_mbs):
            model_input = torch.cat((query, response), axis=1)

            # computing q(x) and P(x)/q(x)
            P_over_q, log_P, q_logprobs, _ = self.compute_helper_values(scores, query, response, query_mask, response_mask, game_data, needed_for_kl=True)
            log_q_x = q_logprobs.cpu().numpy().tolist()
            P_over_q = P_over_q.cpu().numpy().tolist()
            log_P = log_P.detach().cpu().numpy().tolist()

            # if pi_theta is none then calculate kl(q,p)
            # else calculate kl(pi_theta,p)
            if pi_theta is False:
                log_pi_x = log_q_x
            else:
                # Computing π(x)
                query = query.to(self.params["gpt2_device"])
                response = response.to(self.params["gpt2_device"])
                query_mask = query_mask.to(self.params["gpt2_device"])
                response_mask = response_mask.to(self.params["gpt2_device"])
                logits, _, _ = self.model(query, response, query_mask)
                # step1: calculating log prob of the policy pi_theta on
                # sampled generations from q theta
                pi_logprob = logprobs_from_logits(logits[:, :-1], response[:, 1:], mask=response_mask[:, 1:])
                # summing log pi per time step
                pi_logprob = torch.sum(pi_logprob, dim=-1)  # [Batch size]
                log_pi_x = pi_logprob.detach().cpu().numpy().tolist()

            for i, (lg_P, lg_pi, P_q, lg_q) in enumerate(zip(log_P, log_pi_x,
                                                        P_over_q, log_q_x)):
                # calculate the lhs part of the kl equation
                # if P  = 0 , P_q = 0 and lg_p = -inf  the whole lhs = 0 x log(0) = 0
                if P_q == 0 and lg_P == -float("inf"):
                    kl_lhs = 0.0
                else:
                    kl_lhs = P_q * (lg_P - lg_pi) if not cross_entropy else P_q * lg_P
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
        query_mbs = sample_dict['query']
        response_mbs = sample_dict['response']
        scores_mbs = sample_dict['score']
        query_mask_mbs = sample_dict['query_mask']
        response_mask_mbs = sample_dict['response_mask']
        game_data_mbs = sample_dict['game_data']
        values = []

        # iterate over stored queries, responses, and scores
        for query, response, query_mask, response_mask, scores, game_data in zip(query_mbs, response_mbs, query_mask_mbs, response_mask_mbs, scores_mbs, game_data_mbs):
            P_over_q, _, _, _ = self.compute_helper_values(scores, query, response, query_mask, response_mask, game_data)
            P_over_q = P_over_q.detach().cpu().numpy().tolist()
            for P_q in P_over_q:
                values.append(np.abs(1 - P_q / Z_estimate))

        return 0.5 * np.mean(values)

    def compute_tvd_p_pi(self, sample_dict, Z_estimate):
        """
        Computes ||p - π||tvd = 0.5* E x~q(x) |π(x)/q(x) - P(x)/Zq(x)|
        """
        query_mbs = sample_dict['query']
        response_mbs = sample_dict['response']
        scores_mbs = sample_dict['score']
        query_mask_mbs = sample_dict['query_mask']
        response_mask_mbs = sample_dict['response_mask']
        game_data_mbs = sample_dict['game_data']
        values = []

        for query, response, query_mask, response_mask, scores, game_data in zip(query_mbs, response_mbs, query_mask_mbs, response_mask_mbs, scores_mbs, game_data_mbs):
            # Computing π(x)
            query = query.to(self.params["gpt2_device"])
            response = response.to(self.params["gpt2_device"])
            query_mask = query_mask.to(self.params["gpt2_device"])
            response_mask = response_mask.to(self.params["gpt2_device"])
            logits, _, _ = self.model(query, response, query_mask)

            # step1: calculating log prob of the policy pi_theta on sampled generations from q theta
            pi_logprob = logprobs_from_logits(logits[:, :-1], response[:, 1:], mask=response_mask[:, 1:])

            # summing log pi per time step
            pi_logprob = torch.sum(pi_logprob, dim=-1) # [Batch size]
            log_pi_x = pi_logprob.detach().cpu().numpy().tolist()
            P_over_q, _, q_logprobs, _ = self.compute_helper_values(scores, query, response, query_mask, response_mask, game_data)
            log_q_x = q_logprobs.detach().cpu().numpy().tolist()
            P_over_q = P_over_q.detach().cpu().numpy().tolist()

            for i, (lg_pi, P_q, lg_q) in enumerate(zip(log_pi_x, P_over_q, log_q_x)):
                values.append(np.abs(np.exp(lg_pi - lg_q) - P_q / Z_estimate))

        return 0.5 * np.mean(values)

    def compute_helper_values(self, scores, query, response, query_mask, response_mask, game_data, needed_for_kl=False):
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
        query = query.to(next(self.orig_model.parameters()).device)
        response = response.to(next(self.orig_model.parameters()).device)
        query_mask = query_mask.to(next(self.orig_model.parameters()).device)
        response_mask = response_mask.to(next(self.orig_model.parameters()).device)
        orig_logits, _ , _ = self.orig_model(query, response, query_mask)
        orig_logprob = logprobs_from_logits(orig_logits[:, :-1], response[:, 1:], mask=response_mask[:, 1:])

        # calculate b(x)
        orig_logprob = orig_logprob.detach() # we don't backprob to original model
        orig_logprob = torch.sum(orig_logprob, dim=-1) # log(a(x)) of shape [batchsize]

        assert scores.shape == orig_logprob.shape

        # we move all variables to the gpu of the policy to be trained "gpt2_device"
        scores = scores.to(self.params["gpt2_ref_device"])
        orig_logprob = orig_logprob.to(self.params["gpt2_ref_device"])
        scores_ = torch.log(scores.detach()).double()
        log_P = orig_logprob.detach() + scores_  # log P(x) = log a(x) + log(scores)

        # step2: calculate q(x)

        query = query.to(next(self.ref_model.parameters()).device)
        response = response.to(next(self.ref_model.parameters()).device)
        query_mask = query_mask.to(next(self.ref_model.parameters()).device)
        response_mask = response_mask.to(next(self.ref_model.parameters()).device)
        ref_logits, _ , _ = self.ref_model(query, response, query_mask)

        q_logprobs = logprobs_from_logits(ref_logits[:, :-1], response[:, 1:], mask=response_mask[:, 1:])
        q_logprobs = q_logprobs.detach()  # do not backpropagate to q(x)

        q_logprobs = torch.sum(q_logprobs, dim=-1) # Log(q(x)) [Batch size]

        # final reward = exp(Log(P(x)) - Log(q(x)))
        P_over_q = torch.exp(log_P - q_logprobs.to(log_P.device))

        return P_over_q, log_P, q_logprobs, orig_logprob  # R(x) ,P(x), q(x), a(x)

    def prepare_checkpoint(self):

        ckpt = {
                'iter': self.iter,
                'model_state': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
            }

        return ckpt

    def eval_kl_a(self, force_condition=None):

        """
        Computes and returns KL divergence with the original unmodified language model (i.e GPT2)
        """
        sample_dict = self.build_samples_buffer(model = self.get_eval_model(), n_samples = self.params['batch_size'], force_condition=force_condition)

        query_mbs = sample_dict['query']
        response_mbs = sample_dict['response']
        scores_mbs = sample_dict['score']
        query_mask_mb = sample_dict['query_mask']
        response_mask_mb = sample_dict['response_mask']
        game_data_mb = sample_dict['game_data']
        values = []

        eval_model = self.get_eval_model()
        eval_model_device = next(eval_model.parameters()).device

        for query, response, query_mask, response_mask, scores, game_data in zip(query_mbs, response_mbs, query_mask_mb, response_mask_mb, scores_mbs, game_data_mb):
            # Computing eval_model(x)
            logits, _, _ = eval_model(query, response, query_mask)

            ## step1: calculating log prob of the policy pi_theta on sampled generations from q theta
            model_logprob = logprobs_from_logits(logits[:, :-1], response[:, 1:], mask=response_mask[:, 1:])

            # summing log pi per time step
            model_logprob = torch.sum(model_logprob, dim=-1) # [Batch size]

            _, _, _, orig_logprob = self.compute_helper_values(scores, query, response, query_mask, response_mask, game_data, needed_for_kl=True)
            orig_logprob= orig_logprob

            values.append(torch.mean(model_logprob - orig_logprob.to(eval_model_device)).item())

        return np.mean(values)


    def eval_kl_p(self, force_condition=None, cross_entropy=False):

        """
        Compute KL(P || .) on a batch of samples
        samples come from reference model
        """
        # compute KL(P || pi)
        num_samples = self.params.get('num_samples_for_KL')
        num_samples_per_query = self.params.get('num_samples_per_query_for_KL')
        num_queries = int(num_samples / num_samples_per_query)
        assert num_samples_per_query >= self.params['forward_batch_size']
        assert num_samples_per_query % self.params['forward_batch_size'] == 0
        assert num_samples >= self.params['batch_size']
        sample_dict = self.build_samples_buffer(model=self.ref_model, n_samples=num_samples, force_condition=force_condition)
        num_minibatches_per_query = int(num_samples_per_query / self.params['forward_batch_size'])
        print(f'Computing Eval/KL {num_queries} queries with {num_samples_per_query} samples per each')
        kls = []
        for i in range(num_queries):
            idx = slice(i * num_minibatches_per_query, (i + 1) * num_minibatches_per_query)
            query_sample_dict = {}
            for key, value in sample_dict.items():
                if key == 'game_data':
                    assert np.unique(sample_dict['game_data'][idx.start]['Z_c']).shape == (
                    1,), 'Z_cs in az minibatch must be all the same!'
                    assert np.unique(sample_dict['game_data'][idx.stop - 1]['Z_c']).shape == (
                    1,), 'Z_cs in a minibatch must be all the same!'
                    Z_c_i = sample_dict['game_data'][idx.stop - 1]['Z_c'][0] + 1e-8
                    query_sample_dict[key] = value[idx]
                else:
                    query_sample_dict[key] = value[idx]
            kls.append(self.compute_kl(query_sample_dict, Z_c_i, pi_theta=True, cross_entropy=cross_entropy))
        return sum(kls)/num_queries

    @abc.abstractmethod
    def save_checkpoint(self):
        pass

    @abc.abstractmethod
    def load_checkpoint(self):
        pass

    def compute_Z_c_using_MC(self, query_tensors, response_tensors, query_masks, response_masks, game_data, scores, num_samples_per_query):
        sample_size = query_tensors.size(0)
        num_queries = int(sample_size/num_samples_per_query)
        sampling_model = self.get_sampling_model()
        Z_cs = []
        for i in range(num_queries):
            indices_for_query = slice(i * num_samples_per_query, (i+1) * num_samples_per_query)
            importance_weight_minibatches = []
            for start_idx in range(indices_for_query.start, indices_for_query.stop, self.params['forward_batch_size']):
                # if num_samples_per_query > forward_batch_size, divide into minbatches; single minibatch otherwise
                idx = slice(start_idx, start_idx + min(self.params['forward_batch_size'], num_samples_per_query))
                with torch.no_grad():
                    logits, _, _ = sampling_model(
                    query=query_tensors[idx],
                    response=response_tensors[idx],
                    query_mask=query_masks[idx]
                    )
                    log_probs = logprobs_from_logits(
                        logits[:, :-1],
                        response_tensors[idx, 1:],
                        mask=response_masks[idx, 1:]
                    ).sum(dim=1)
                    orig_logits, _, _ = self.orig_model(
                        query=query_tensors[idx].to(self.params['gpt2_orig_device']),
                        response=response_tensors[idx].to(self.params['gpt2_orig_device']),
                        query_mask=query_masks[idx].to(self.params['gpt2_orig_device'])
                    )
                    orig_log_probs = logprobs_from_logits(
                        orig_logits[:, :-1],
                        response_tensors[idx, 1:].to(self.params['gpt2_orig_device']),
                        mask=response_masks[idx, 1:].to(self.params['gpt2_orig_device'])
                    ).sum(dim=1).to(self.params['gpt2_device'])
                importance_weight = torch.exp(orig_log_probs - log_probs).detach()
                importance_weight_minibatches.append(importance_weight)
            importance_weight_c = torch.cat(importance_weight_minibatches)
            scores_c = scores[indices_for_query].to(self.params["gpt2_device"]) + 1e-4
            Z_c_i = (scores_c * importance_weight_c).mean().item()
            Z_cs += [Z_c_i] * num_samples_per_query
        game_data['Z_c'] = Z_cs
