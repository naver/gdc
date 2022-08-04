
import ast
import abc
import io
import os
import contextlib
from itertools import takewhile
import json
from codeop import compile_command

import nltk
from nltk import ngrams
import spacy

from nltk.translate.bleu_score import SmoothingFunction
try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords')
from nltk.corpus import stopwords
from transformers import GPT2Tokenizer, pipeline
import datasets
from .core import *
import torch
import pycodestyle


class Metric:
    """
    Defines a text quality metric.
    """

    def get_name(self):
        return self.name

    @abc.abstractmethod
    def compute_metric(self, response, reference=None, query=None, **kwargs):
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
        self.referenceless = True

    def compute_metric(self, response, reference=None, query=None, **kwargs):
        return self._distinct_ngrams(response, self.n)

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
        self.referenceless = True

    def compute_metric(self, response, reference=None, query=None, **kwargs):

        self.reference = response
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

    def _calc_bleu(self, reference, hypothesis, weight):
        return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                       smoothing_function=SmoothingFunction().method1)

    def _get_bleu_fast(self):
        reference = self._get_reference()
        # random.shuffle(reference)
        reference = reference[0:self.sample_size]
        return self._get_bleu(reference=reference)

    def _get_bleu(self, reference=None):
        ngram = self.gram
        if reference is None:
            reference = self._get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        scores = list()
        sentence_num = len(reference)
        for index in range(sentence_num):
            hypothesis = reference[index]
            other = reference[:index] + reference[index+1:]
            scores.append(self._calc_bleu(other, hypothesis, weight))
        return sum(scores)/len(scores)


class SacreBLEU(Metric):

    def __init__(self):
        self.name = "BLEU"
        self.sacrebleu = datasets.load_metric('sacrebleu')
        self.referenceless = False

    def compute_metric(self, response, reference=None, query=None, **kwargs):
        output = self.sacrebleu.compute(
            predictions=response,
            references=[[r] for r in reference]
        )
        return output['score']


class Rouge(Metric):

    def __init__(self):
        self.name = 'Rouge'
        self.rouge = datasets.load_metric('rouge')
        self.referenceless = False

    def compute_metric(self, response, reference=None, query=None, **kwargs):
        output = self.rouge.compute(
            predictions=response,
            references=reference
        )
        response_dict = {}
        for name, value in output.items():
            response_dict[f'{name}/f1'] = value.mid.fmeasure
            response_dict[f'{name}/precision'] = value.mid.precision
            response_dict[f'{name}/recall'] = value.mid.recall
        return response_dict


class CharLength(Metric):

    name = 'CharLength'
    referenceless = True

    def compute_metric(self, response, reference=None, query=None, **kwargs):
        return sum(len(r) for r in response)/len(response)


class TokenLength(Metric):

    def __init__(self, tokenizer):
        self.name = 'TokenLength'
        self.tokenizer = tokenizer
        self.referenceless = True

    def compute_metric(self, response, reference=None, query=None, **kwargs):
        retokenized_responses = self.tokenizer(response, return_tensors='pt', padding=True)['input_ids']
        return torch.count_nonzero(retokenized_responses, dim=1).float().mean().item()


class WordLength(Metric):

    def __init__(self):
        self.name = 'WordLength'
        self.spacy_tokenizer = spacy.blank('en').tokenizer
        self.referenceless = True

    def compute_metric(self, response, reference=None, query=None, **kwargs):
        tokenized_responses = list(self.spacy_tokenizer.pipe(response, batch_size=128))
        return sum(len(response) for response in tokenized_responses)/len(tokenized_responses)


class BertScore(Metric):

    def __init__(self, language='en', batch_size=32, device=0, limit_samples=16):
        self.name = 'BertScore'
        self.bert_score = datasets.load_metric('bertscore')
        self.language = language
        self.batch_size = batch_size
        self.device = device
        self.limit_samples = limit_samples
        self.referenceless = False

    def compute_metric(self, response, reference=None, query=None, **kwargs):
        output = self.bert_score.compute(
            predictions=response[:self.limit_samples],
            references=reference[:self.limit_samples],
            lang=self.language,
            model_type='roberta-base',
            num_layers=10,
            batch_size=self.batch_size,
            device=self.device
        )
        return {f'BertScore_{k}': sum(v)/len(v) for k, v in output.items() if k != 'hashcode'}


class BLEURT(Metric):

    def __init__(self):
        self.name = 'BLEURT'
        self.bleurt = datasets.load_metric('bleurt')
        self.referenceless = False

    def compute_metric(self, response, reference=None, query=None, **kwargs):
        scores = self.bleurt.compute(
            predictions=response,
            references=reference,
        )['scores']
        return sum(scores)/len(scores)


class NamedEntities(Metric):

    def __init__(self):
        self.spacy_model = self.spacy_model = spacy.load(
                "en_core_web_sm",
                disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"]
            )
        self.REQUIRED_NERS = ['PERSON', 'FAC', 'GPE', 'ORG', 'NORP', 'LOC', 'EVENT']
        self.referenceless = False

    def compute_metric(self, response, reference=None, query=None, **kwargs):
        queries_processed = self.spacy_model.pipe(query, batch_size=128)
        responses_processed = self.spacy_model.pipe(response, batch_size=128)
        references_processed = self.spacy_model.pipe(reference, batch_size=128)
        source_precisions, target_precisions, target_recalls, num_ners_summary, num_ners_reference = [], [], [], [], []
        named_entities_in_both_summary_and_target = []
        named_entities_in_both_target_and_source = []
        for query, response, reference in zip(queries_processed, responses_processed, references_processed):
            ners_in_response = set(span.text.lower() for span in response.ents if span.label_ in self.REQUIRED_NERS)
            ners_in_reference = set(span.text.lower() for span in reference.ents if span.label_ in self.REQUIRED_NERS)
            ners_in_query = set(span.text.lower() for span in query.ents if span.label_ in self.REQUIRED_NERS)
            if len(ners_in_response) > 0:
                source_precisions.append(len(ners_in_response & ners_in_query)/len(ners_in_response))
                target_precisions.append(len(ners_in_response & ners_in_reference) / len(ners_in_response))
            else:
                source_precisions.append(1)
                target_precisions.append(1)
            if len(ners_in_reference) > 0:
                target_recalls.append(len(ners_in_response & ners_in_reference) / len(ners_in_reference))
            else:
                target_recalls.append(1)
            num_ners_summary.append(len(ners_in_response))
            num_ners_reference.append(len(ners_in_reference))
            named_entities_in_both_summary_and_target.append(len(ners_in_response & ners_in_reference))
            named_entities_in_both_target_and_source.append(len(ners_in_query & ners_in_reference))
        return {
            'named_entities_precision_source': sum(source_precisions)/len(source_precisions),
            'named_entities_precision_target': sum(target_precisions)/len(target_precisions),
            'named_entities_recall_target': sum(target_recalls)/len(target_recalls),
            'named_entities_in_summary': sum(num_ners_summary)/len(num_ners_summary),
            'named_entities_in_reference': sum(num_ners_reference)/len(num_ners_reference),
            'named_entities_in_both_summary_and_target': sum(named_entities_in_both_summary_and_target) / len(named_entities_in_both_summary_and_target),
            'named_entities_in_both_target_and_source': sum(named_entities_in_both_target_and_source)/len(named_entities_in_both_target_and_source)
        }


class ASTNodeCount(Metric):

    def __init__(self):
        self.name = 'AST node count'
        self.referenceless = True

    def compute_metric(self, response, reference=None, query=None, **kwargs):
        counts = []
        for signature, body in zip(query, response):
            try:
                tree = ast.parse(signature + body, mode='exec')
                node_count = len(list(ast.walk(tree)))
                counts.append(node_count)
            except:
                pass
        if len(counts) > 0:
            return sum(counts)/len(counts)
        else:
            return float('nan')


class PEP8Errors(Metric):

    def __init__(self):
        self.name = 'Code quality'
        self.referenceless = True

    def compute_metric(self, response, reference=None, query=None, **kwargs):
        error_counts = []
        for signature, body in zip(query, response):
            virtual_file = io.StringIO(signature + body)
            checker = pycodestyle.Checker(lines=virtual_file.readlines(), show_source=True)
            with contextlib.redirect_stdout(open(os.devnull, 'w')):  # keep stdout clean
                try:
                    num_errors = checker.check_all()
                    error_counts.append(num_errors)
                except (UnicodeEncodeError, IndexError):
                    pass
        return sum(error_counts)/len(error_counts)


class UpvoteScore(Metric):

    def __init__(self, device=-1):
        self.name = 'Upvote score'
        self.referenceless = True
        self.scorer = pipeline('text-classification', 'microsoft/DialogRPT-updown', device=device)

    def compute_metric(self, response, reference=None, query=None, **kwargs):
        inputs = [f'{q}<|endoftext|>{r}' for q, r in zip(query, response)]
        scores = [pred['score'] for pred in self.scorer(inputs)]
        return sum(scores)/len(scores)


class Repetitions(Metric):

    def __init__(self, n=1):
        self.name = f'Repeated_{n}-grams'
        self.referenceless = True
        self.n = n
        self.stopwords = stopwords.words('english')

    def compute_metric(self, response, reference=None, query=None, **kwargs):
        scores = []
        for q, r in zip(query, response):
            try:
                scores.append(self._compute_repetitions_single(q, r))
            except ZeroDivisionError:
                continue
        if len(scores) > 0:
            return sum(scores)/len(scores)
        else:
            return float('nan')

    def _compute_repetitions_single(self, query, response):
        response_tokens = nltk.tokenize.word_tokenize(response)
        response_ngrams = set(' '.join(response_tokens[i:i + self.n]).lower()
                              for i in range(len(response_tokens) - (self.n-1)))
        if self.n == 1:
            response_ngrams = set(word for word in response_ngrams if word not in self.stopwords)
        query_tokens = nltk.tokenize.word_tokenize(query)
        query_ngrams = set(' '.join(query_tokens[i:i + self.n]).lower()
                           for i in range(len(query_tokens) - (self.n-1)))
        if self.n == 1:
            query_ngrams = set(word for word in query_ngrams if word not in self.stopwords)
        return len(response_ngrams & query_ngrams)/len(response_ngrams)


class Specificity(Metric):

    def __init__(self, word2idf_path):
        self.name = 'Specificity'
        self.referenceless = True
        self.worf2idf = json.load(open(word2idf_path, encoding='utf-8'))
        self.min_idf = min(self.worf2idf.values())
        self.max_idf = max(self.worf2idf.values())

    def compute_metric(self, response, reference=None, query=None, **kwargs):
        scores = []
        for r in response:
            token_idfs = [self.worf2idf.get(token, self.max_idf) for token in nltk.tokenize.word_tokenize(r)]
            token_nidfs = [(idf - self.min_idf)/(self.max_idf - self.min_idf) for idf in token_idfs]
            try:
                scores.append(sum(token_nidfs)/len(token_nidfs))
            except ZeroDivisionError:
                continue
        return sum(scores)/len(scores)


class Compilability(Metric):

    def __init__(self):
        self.name = 'Compilability'
        self.referenceless = True

    def compute_metric(self, response, reference=None, query=None, **kwargs):
        scores = []
        for signature, body in zip(query, response):
            try:
                if compile_command(signature + body, symbol='exec'):
                    scores.append(1)
                else:  # sample is a prefix of valid Python code but is is not valid Python code itself
                    scores.append(0)
            except:
                scores.append(0)
        return sum(scores)/len(scores)
