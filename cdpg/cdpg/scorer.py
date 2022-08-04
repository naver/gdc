import os
from codeop import compile_command
import io
import contextlib

import torch
import nltk
try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')

import pycodestyle


REQUIRED_NERS = ['PERSON', 'FAC', 'GPE', 'ORG', 'NORP', 'LOC', 'EVENT']


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
        self.POSITIVE = 1
        self.NEGATIVE = 0
        assert config['scorer_type'] in ['phrase_table', 'named_entities', 'compiler',  'pep8'], \
                            "incorrect scoring type {}".format(config['scorer_type'])

    def get_scoring_fn(self):
        """
        returns a function that takes a list of strings, scores each one, and returns a torch tensor of scores
        """

        if self.config["scorer_type"] == "phrase_table":

            phrase_table = self.config["scorer_attribute"]

            def single_score_fn(continuation, query=None):
                for term in phrase_table:
                    if term in query and phrase_table[term] not in continuation:
                        return self.NEGATIVE
                return self.POSITIVE

            def scoring_fn(continuations, queries=None, **kwargs):
                if queries is None:
                    return torch.tensor([single_score_fn(s) for s in continuations])
                else:
                    return torch.tensor([single_score_fn(continuation=continuation, query=query)
                                         for query, continuation in zip(queries, continuations)])

        elif self.config["scorer_type"] == "named_entities":

            def scoring_fn(continuations, queries=None, **kwargs):
                num_entities_required = int(self.config.get('scorer_attribute', 2))
                queries_processed = self.spacy_model.pipe(queries, batch_size=128)
                continuations_processed = self.spacy_model.pipe(continuations, batch_size=128)
                scores = []
                for query, continuation in zip(queries_processed, continuations_processed):
                    ners_in_query = set(span.text.lower() for span in query.ents if span.label_ in REQUIRED_NERS)
                    ners_in_continuation = set(span.text.lower() for span in continuation.ents if span.label_ in REQUIRED_NERS)
                    if len(ners_in_continuation) >= num_entities_required and ners_in_continuation.issubset(ners_in_query):
                        scores.append(self.POSITIVE)
                    else:
                        scores.append(self.NEGATIVE)
                return torch.tensor(scores)

        elif self.config['scorer_type'] == 'compiler':

            partial = self.config.get('scorer_attribute', '') == 'partial'

            def single_score_fn(signature, body):
                try:
                    if compile_command(signature+body, symbol='exec'):
                        return self.POSITIVE
                    else:  # sample is a prefix of valid Python code but is is not valid Python code itself
                        return self.NEGATIVE if not partial else self.POSITIVE
                except:
                    return self.NEGATIVE

            def scoring_fn(continuations, queries=None, **kwargs):
                return torch.tensor([single_score_fn(signature, body) for signature, body in zip(queries, continuations)])

        elif self.config['scorer_type'] == 'pep8':

            def single_score_fn(signature, body):
                virtual_file = io.StringIO(signature + body + '\n')
                checker = pycodestyle.Checker(lines=virtual_file.readlines(), show_source=True)
                with contextlib.redirect_stdout(open(os.devnull, 'w')):  # keep stdout clean
                    try:
                        if checker.check_all() == 0:
                            return self.POSITIVE
                        else:
                            return self.NEGATIVE
                    except (UnicodeEncodeError, IndexError):
                        return self.NEGATIVE

            def scoring_fn(continuations, queries=None, **kwargs):
                return torch.tensor(
                    [single_score_fn(signature, body) for signature, body in zip(queries, continuations)])

        return scoring_fn
