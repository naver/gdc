# GDC
# Copyright 2021-present NAVER Corp.
# Distributed under CC-BY-NC-SA-4.0

__all__ = ['Metric', 'Distinct_N', 'SelfBlEU']


import numpy as np
import collections
import time
import random
import copy

import nltk
from nltk import ngrams

import os
from multiprocessing import Pool
from nltk.translate.bleu_score import SmoothingFunction
import abc
from .gpt2 import GPT2HeadWithValueModel, respond_to_batch
from transformers import GPT2Tokenizer
from .core import *
import torch


class Metric():
    """
    Defines a text quality metric.
    """

    def get_name(self):
        return self.name

    @abc.abstractmethod
    def compute_metric(self, texts):
        pass


class Distinct_N(Metric):

    def __init__(self, n):
        """
        Distinct n-grams metrics. This is a sequence-level diversity metric.
        See https://www.aclweb.org/anthology/N16-1014 for more details.

        Args:
            n (int): n-grams 
        """

        self.n = n
        self.name = f'Distinct_{n}'

    def compute_metric(self, texts):
        return self._distinct_ngrams(texts, self.n)

    def _distinct_ngrams(self, texts, n):
        total = 0.0
        for t in texts:
            try:
                tokens = nltk.tokenize.word_tokenize(t)
                n_distinct = len(set(ngrams(tokens, n)))
                total += n_distinct/ len(tokens)
            except:
                continue

        return total / len(texts)


class SelfBlEU(Metric):

    def __init__(self, gram=3, sample_size=500):
        """
        Corpus level diversity metric. See https://arxiv.org/abs/1802.01886 for more details.
        """
        super().__init__()
        self.name = 'Self-BLEU-' + str(gram)
        self.gram = gram
        self.sample_size = sample_size
        self.reference = None
        self.is_first = True

    def compute_metric(self, texts):

        self.reference = texts
        return self._get_bleu_fast()


    def _get_score(self, is_fast=True, ignore=False):
        if ignore:
            return 0
        if self.is_first:
            self.get_reference()
            self.is_first = False
        if is_fast:
            return self._get_bleu_fast()
        return self._get_bleu_parallel()

    def _get_reference(self):
        if self.reference is None:
            self.reference = self.test_data
            return self.reference
        else:
            return self.reference

    def _get_bleu(self):
        ngram = self.gram
        bleu = list()
        reference = self._get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))

        for hypothesis in test_data:
            hypothesis = nltk.word_tokenize(hypothesis)
            bleu.append(nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                                smoothing_function=SmoothingFunction().method1))
        return sum(bleu) / len(bleu)

    def _calc_bleu(self, reference, hypothesis, weight):
        return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                       smoothing_function=SmoothingFunction().method1)

    def _get_bleu_fast(self):
        reference = self._get_reference()
        # random.shuffle(reference)
        reference = reference[0:self.sample_size]
        return self._get_bleu_parallel(reference=reference)

    def _get_bleu_parallel(self, reference=None):
        ngram = self.gram
        if reference is None:
            reference = self._get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(os.cpu_count())
        result = list()
        sentence_num = len(reference)
        for index in range(sentence_num):
            hypothesis = reference[index]
            other = reference[:index] + reference[index+1:]
            result.append(pool.apply_async(self._calc_bleu, args=(other, hypothesis, weight)))

        score = 0.0
        cnt = 0
        for i in result:
            score += i.get()
            cnt += 1
        pool.close()
        pool.join()
        return score / cnt
#exports

class GPT2_score(Metric):

    def __init__(self, device=3):
        """
        This is the log probability according to the original GPT2 model 
        """

        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2HeadWithValueModel.from_pretrained('gpt2').to(device)
        self.model.eval()
        self.name= "GPT2-Score"

    
    def compute_metric(self, samples):

        tokenized = build_gpt2_batch_from_txt(samples, self.tokenizer, self.device)
        fbs = 32
        if len(samples) < fbs:
            fbs = len(samples)

        sample_size = len(samples)
        all_logprob = [] 

        for i in range(int(sample_size/fbs)):
            cur_batch = tokenized[i*fbs:(i+1)*fbs]
            orig_logits, _ , _ = self.model(cur_batch)
            orig_logprob = logprobs_from_logits(orig_logits[:,:-1,:], cur_batch[:, 1:])
            mask = (cur_batch[:, 1:] != self.tokenizer.bos_token_id).float()
            
            orig_logprob = orig_logprob * mask 

            # calculate b(x)
            orig_logprob = orig_logprob.detach() # we don't backprob to original model
            orig_logprob = torch.sum(orig_logprob, dim=-1) # log(a(x)) of shape [batchsize]
            all_logprob.append(orig_logprob)
        
        all_logprob = torch.cat(all_logprob)
        return all_logprob.mean().item()
