# GDC
# Copyright 2021-present NAVER Corp.
# Distributed under CC-BY-NC-SA-4.0

#! /usr/bin/env python3
# coding=utf-8

import os
import sys

import numpy as np

import torch
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)

lab_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..')
sys.path.insert(1, lab_root)

class ClassificationHead(torch.nn.Module):
    """ Language Model Head for the transformer """

    def __init__(self, class_size=5, embed_size=2048):
        super(ClassificationHead, self).__init__()
        self.class_size = class_size
        self.embed_size = embed_size
        self.mlp = (torch.nn.Linear(embed_size, class_size))

    def forward(self, hidden_state):
        lm_logits = self.mlp(hidden_state)
        return lm_logits


class Discriminator2mean(torch.nn.Module):
    def __init__(self, model, class_size=5, embed_size=1024, device=1):
        super(Discriminator2mean, self).__init__()
        self.classifierhead = ClassificationHead(class_size=class_size, embed_size=embed_size)
        self.model = model
        self.embed_size = embed_size
        self.device=device

    def get_classifier(self):
        return self.classifierhead

    def train_custom(self):
        for param in self.model.parameters():
            param.requires_grad = False
        pass
        self.classifierhead.train()

    def forward(self, x):
        mask_src = 1 - x.eq(0).unsqueeze(1).type(torch.FloatTensor).to(self.device).detach()
        mask_src = mask_src.repeat(1, self.embed_size, 1)
        x = self.forward_embed(x)
        hidden, x = self.forward_transformer_embed(x)
        #  Hidden has shape batch_size x length x embed-dim

        hidden = hidden.permute(0, 2, 1)
        _, _, batch_length = hidden.shape
        hidden = hidden * mask_src
        #
        hidden = hidden.permute(0, 2, 1)
        x = torch.sum(hidden, dim=1)/(torch.sum(mask_src, dim=-1).detach() + 1e-10)
        x = self.classifierhead(x)
        x = F.softmax(x, dim=-1)
        return x

    # def forward_embed(self, inputs_ids, position_ids=None, token_type_ids=None, past=None):
    #     """
    #     This is kind of a hack imported from PPLM
    #     https://github.com/uber-research/PPLM/blob/master/paper_code/pytorch_pretrained_bert/modeling_gpt2.py#L750
    #     We replicate it here to be able to use their pretrained classifiers 
    #     which is a feed fwd NN on top of embeddings of GPT2

    #     Args:
    #         inputs_ids (Tensor): batch_size x length x embed-dim

    #     Returns:
    #         Tensor: batch_size x length x embed-dim
    #     """
    #     hidden_states = self.model.transformer_forward_embed(inputs_ids)
    #     return hidden_states
    


        # The following methods are a kind of a hack imported from PPLM
        # https://github.com/uber-research/PPLM/blob/master/paper_code/pytorch_pretrained_bert/modeling_gpt2.py#L750
        # We replicate it here to be able to use their pretrained classifiers
        # which is a feed fwd NN on top of embeddings of GPT2
    
    def forward_transformer(self, hidden_states, past=None, add_one=False):
        transformer = self.model.transformer
        if past is None:
            past = [None] * len(transformer.h)
        presents = []
        hiddens = []
        for block, layer_past in zip(transformer.h, past):
            hidden_states, present = block(hidden_states, layer_past)
            hiddens.append(hidden_states)
            presents.append(present)
        hidden_states = transformer.ln_f(hidden_states)
        transformer.hiddens_list = hiddens
        transformer.hidden_states = hidden_states
        if add_one:
            output_shape = (transformer.input_shape[0],) + (transformer.input_shape[1] + 1,) + (hidden_states.size(-1),)
        else:
            output_shape = (transformer.input_shape[0],) + (transformer.input_shape[1],) + (hidden_states.size(-1),)

        if add_one:
            present_shape = (transformer.input_shape[0],) + (transformer.input_shape[1] + 1,) + (2*hidden_states.size(-1),)
        else:
            present_shape = (transformer.input_shape[0],) + (transformer.input_shape[1],) + (2*hidden_states.size(-1),)

        presents = [p.view(*present_shape) for p in presents]

        return hidden_states.view(*output_shape), presents


    def forward_transformer_embed(self, hidden_states, past=None, add_one=False):
        hidden_states, presents = self.forward_transformer(hidden_states,
                                                                        past, add_one=add_one)
        return hidden_states, presents


    def forward_embed(self, input_ids, position_ids=None, token_type_ids=None, past=None):
        gpt2 = self.model.transformer
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()

        if past is None:
            past_length = 0
            past = [None] * len(gpt2.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        gpt2.input_shape = input_ids.size()

        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = gpt2.wte(input_ids)
        #inputs_embeds.retain_grad()
        gpt2.i_embeds = inputs_embeds
        position_embeds = gpt2.wpe(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = gpt2.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        ### HACK by PPLM avoid the last layer and output the hidden state
        # presents = []
        # for block, layer_past in zip(self.h, past):
        #     hidden_states, present = block(hidden_states, layer_past)
        #     presents.append(present)
        # hidden_states = self.ln_f(hidden_states)
        #
        # output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states