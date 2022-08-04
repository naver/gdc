import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model, GPT2PreTrainedModel, T5ForConditionalGeneration, \
    GPT2Config, GPTNeoForCausalLM, LogitsProcessor, LogitsProcessorList, StoppingCriteriaList, MaxLengthCriteria
try:
    from transformers.modeling_utils import top_k_top_p_filtering
except:
    from transformers import top_k_top_p_filtering


class GPT2Wrapper(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weighted_decoding = False

    @classmethod
    def from_pretrained(cls, *args):
        instance = cls()
        instance.add_module(
            'model',
            GPT2LMHeadModel.from_pretrained(*args, resid_pdrop=0, attn_pdrop=0, summary_first_dropout=0, embd_pdrop=0)
        )
        return instance

    def get_prefix(self, task_name):
        if task_name == 'dialogue':
            return ''
        elif task_name == 'code_generation_old':
            return '<s>'

    def forward(self, query, response, query_mask, *args, **kwargs):
        if response is None:
            input_ids = query
            attention_mask = query_mask
            position_ids = None
        else:
            input_ids = torch.cat((query, response), axis=1)
            response_mask = (response != self.model.config.pad_token_id).long()
            attention_mask = torch.cat((query_mask, response_mask), axis=1)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids
        ).logits
        if response is None:
            lm_logits = logits
        else:
            lm_logits = logits[:, query.size(1):]
        return lm_logits, None, None

    def respond_to_batch(self, input_ids, attention_mask, txt_len=20, top_k=0, top_p=0.9, decoding_params=None):
        decoding_params = decoding_params.copy()
        decoding_params['max_length'] += input_ids.size(1)
        response = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **decoding_params
        )

        def correct_shape(response):
            """ response is on the form (bsz, txt_len)
                but sometimes shorted if all batch gen sequences are short
                that could cause some errors in merging batches later
            """
            max_length = decoding_params['max_length']
            if response.shape[-1] < max_length:
                # create tensor of correct shape with padding
                r = response[0].data.new(response.shape[0], max_length).fill_(self.model.config.pad_token_id)
                # fill in the values
                r[:, :response.shape[1]] = response
                return r
            else:
                return response

        response = correct_shape(response)

        response_mask = (response != self.model.config.pad_token_id).long()
        return response[:, input_ids.size(1):], response_mask[:, input_ids.size(1):]


class GPTNeoWrapper(GPT2Wrapper):

    def get_prefix(self, task_name):
        return ''

    @classmethod
    def from_pretrained(cls, *args):
        instance = cls()
        instance.add_module(
            'model',
            GPTNeoForCausalLM.from_pretrained(*args, embed_dropout=0, attention_dropout=0, resid_dropout=0)
        )
        return instance


class T5ForConditionalGenerationWrapper(torch.nn.Module):

    @classmethod
    def from_pretrained(cls, *args):
        instance = cls()
        dropout = 0
        instance.add_module('model', T5ForConditionalGeneration.from_pretrained(*args, dropout_rate=dropout))
        return instance

    def get_prefix(self, task_name):
        return self.model.config.task_specific_params[task_name]['prefix']

    def forward(self, query, response, query_mask, *args, **kwargs):
        outputs = self.model(
            input_ids=query,
            decoder_input_ids=response,
            attention_mask=query_mask,
            # when set to None, the default behavior to remove padding and allow causal masking
            decoder_attention_mask=None,
            *args, **kwargs
        )
        return outputs.logits, None, None

    def respond_to_batch(self, input_ids, attention_mask, txt_len=20, top_k=0, top_p=0.9, decoding_params=None):
        response = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **decoding_params
        )

        def correct_shape(response):
            """ response is on the form (bsz, txt_len)
                but sometimes shorted if all batch gen sequences are short
                that could cause some errors in merging batches later
            """
            max_txt_len = decoding_params.get('max_length')
            if response.shape[-1] < max_txt_len:
                # create tensor of correct shape with padding
                r = response[0].data.new(response.shape[0], max_txt_len).fill_(self.model.config.pad_token_id)
                # fill in the values
                r[:, :response.shape[1]] = response
                return r
            else:
                return response

        response = correct_shape(response)

        response_mask = (response != self.model.config.pad_token_id).long()
        return response, response_mask
