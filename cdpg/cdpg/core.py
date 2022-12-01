import collections
from itertools import takewhile

import torch
import torch.nn.functional as F
import numpy as np


def flatten_dict(nested, sep='/'):
    """Flatten dictionary and concatenate nested keys with separator."""
    def rec(nest, prefix, into):
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, collections.Mapping):
                rec(v, prefix + k + sep, into)
            else:
                into[prefix + k] = v
    flat = {}
    rec(nested, '', flat)
    return flat


def stack_dicts(stats_dicts):
    """Stack the values of a dict."""
    results = dict()
    for k in stats_dicts[0]:

        stats_list = [torch.flatten(d[k]) for d in stats_dicts]
        results[k] = torch.stack(stats_list)
    return results


def stats_to_np(stats_dict):
    """Cast all torch.tensors in dict to numpy arrays."""
    new_dict = dict()
    for k, v in stats_dict.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = v.detach().cpu().numpy()
        else:
            new_dict[k] = v
        if np.isscalar(new_dict[k]):
            new_dict[k] = float(new_dict[k])
    return new_dict


def logprobs_from_logits(logits, labels, mask=None):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = F.log_softmax(logits, dim=2)
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy * mask if mask is not None else logpy


def entropy_from_logits(logits):
    """Calculate entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, axis=-1) - torch.sum(pd*logits, axis=-1)
    return entropy


def is_line_indented(line):
    return line.startswith(' ' * 4) or line.startswith('\t')


def postprocess_code_generation(response):
    # Remove everything after function declaration ends (which is marked by lack of indentation)
    return '\n'.join(takewhile(is_line_indented, response.split('\n')))
