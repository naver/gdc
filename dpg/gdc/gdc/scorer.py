# GDC
# Copyright 2021-present NAVER Corp.
# Distributed under CC-BY-NC-SA-4.0

__all__ = ['Scorer']



import os
import pathlib

import torch
from .gpt2tunediscrim import Discriminator2mean, ClassificationHead
import nltk
try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')
from nltk.tokenize import word_tokenize as tokenize
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer

from .core import build_gpt2_batch_from_txt


### Finetuned classification heads (taken from https://github.com/uber-research/PPLM/tree/master/paper_code/discrim_models)
map_attribute_to_head= {
    'sentiment':'./resources/discrim_models/sentiment_classifierhead.pt',
    'toxicity': './resources/discrim_models/toxicity_classifierhead.pt',
    'clickbait': './resources/discrim_models/clickbait_classifierhead.pt'
}

### Number of classes for each task
map_attribute_to_class_count = {
    'sentiment': 5,
    'toxicity': 2,
    'clickbait': 2
}


male_pronouns = ["he", "him", "himself"]
female_pronouns = ["she", "her", "herself"]


class Scorer:

    """
    This class defines a scorer that takes a textual input and output a scoring value. 

    """

    def __init__(self, **config):
        """
        Args:
            config (python dict): a dictionary describing scorer attributes. 
            For example, {"type": "single_word" or "wordlist" or "model",
                            "attribute": "amazing" or "politics" or "sentiment",
        """

        print("initializing scorer...")
        self.config = config


        assert config['scorer_type'] in ['single_word', 'wordlist', 'wikibio-wordlist', 'model', 'gender'], \
                            "incorrect scoring type {}".format(config['scorer_type'])
        assert 'scorer_attribute' in config, \
                "you need to specify a word, wordlist or a model for scoring!"
        
        if self.config.get("reverse_signal", False):
            self.POSITIVE=0.0;self.NEGATIVE=1.0
        else:
            self.POSITIVE=1.0;self.NEGATIVE=0.0


        if self.config['scorer_type'] == 'wordlist':
            ### read and save wordlist
            wordlist_path = os.path.join(pathlib.Path(__file__).parent.absolute(),
                                            './resources/wordlists/', 
                                            self.config['scorer_attribute'] + '.txt')
            assert os.path.exists(wordlist_path), "Can't found wordlist at {}".format(wordlist_path)
            
            with open(wordlist_path) as f:
                self.wordlist = f.read().split('\n')
            print("using wordlist with top words", self.wordlist[:5])

        if self.config['scorer_type'] == 'wikibio-wordlist':
            ### read and save wordlist
            wordlist_path = os.path.join(pathlib.Path(__file__).parent.absolute(),
                                            './resources/wikibio-wordlists/', 
                                            self.config['scorer_attribute'] + '.txt')
            assert os.path.exists(wordlist_path), "Can't found wordlist at {}".format(wordlist_path)
            
            with open(wordlist_path) as f:
                self.wordlist = f.read().split('\n')
            print("using wordlist with top words", self.wordlist[:5])

        
        elif self.config['scorer_type'] == 'model':
            
            ## load GPT2 model and tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
            model = GPT2LMHeadModel.from_pretrained('gpt2-medium')

            # initialize discriminator
            discriminator = Discriminator2mean(model=model, 
                        device=config.get('gpt2_descriminator_device', 1))

            sc_attr = self.config['scorer_attribute']
            assert sc_attr in map_attribute_to_head.keys(), "scorer attribute {} not found".format(sc_attr)

            ### load sentiment head
            classifier = ClassificationHead(class_size=map_attribute_to_class_count[sc_attr], embed_size=1024)
            
            # loading discriminator from resources
            p = os.path.join(pathlib.Path(__file__).parent.absolute(), \
                map_attribute_to_head[sc_attr])

            classifier.load_state_dict(torch.load(p))
            discriminator.classifierhead = classifier
            discriminator.eval()

            self.class_idx = self.config['class_index'] # 2 for positive, 3 for negative
            self.discriminator = discriminator.to(config['gpt2_descriminator_device'])

        elif self.config['scorer_type'] == 'gender':
            assert self.config['scorer_attribute'] in ["male", "female", "other"]

    def get_scoring_fn(self):
        """
        returns a function that takes a list of strings, scores each one, and returns a torch tensor of scores
        """

        if self.config['scorer_type'] =='single_word':

            def scoring_fn(samples):
                tokenized_samples = [tokenize(s) for s in samples]
                return torch.tensor([self.POSITIVE if self.config['scorer_attribute'] in tokens else self.NEGATIVE for tokens in tokenized_samples])

        elif self.config['scorer_type'] in ['wordlist', 'wikibio-wordlist']:

                
            def scoring_fn(samples):
                tokenized_samples = [tokenize(s) for s in samples]
                return torch.tensor([self.POSITIVE if any(w in tokens for w in self.wordlist) else self.NEGATIVE for tokens in tokenized_samples])


        elif self.config['scorer_type'] == 'model':
            
            assert not self.config.get("reverse_signal", False), "Reversing signal not implemented for discriminator"
            def scoring_fn(samples):
                padded_tensors = build_gpt2_batch_from_txt(samples, self.tokenizer, self.config['gpt2_descriminator_device'])
                probs = self.discriminator.forward(padded_tensors).detach()
                preds = probs.argmax(dim=-1)

                if type(self.class_idx) is int:
                    ## if class_idx is an int 1
                    return (preds == self.class_idx).float()
                else:
                    ## if class idx is an array e.g. [1,2]
                    #  iterate over all class idx and check if any of the m
                    # matches the clf predictions. i.e. logical OR operation
                    res = (preds == self.class_idx[0]).float()
                    for idx in self.class_idx[1:]:
                        res += (preds == idx).float()
                    return res


        elif self.config["scorer_type"] == "gender":

            def single_score_fn(s):
                s = tokenize(s)
                count_male_pronouns = sum(s.count(p) for p in male_pronouns)
                count_female_pronouns = sum(s.count(p) for p in female_pronouns)

                if count_male_pronouns > count_female_pronouns:
                    return self.POSITIVE if self.config["scorer_attribute"] == "male" else self.NEGATIVE
                if count_female_pronouns > count_male_pronouns:
                    return self.POSITIVE if self.config["scorer_attribute"] == "female" else self.NEGATIVE
                else: ## equal
                    return self.POSITIVE if self.config["scorer_attribute"] == "other" else self.NEGATIVE
                    
            def scoring_fn(samples):
                return torch.tensor([single_score_fn(s) for s in samples])


        return scoring_fn




