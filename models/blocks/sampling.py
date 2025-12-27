import torch
from einops import rearrange
from torch.nn import functional as F

""" Slightly adapted from  https://github.com/karpathy/minGPT/blob/master/mingpt/model.py """


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

# temperature=1.0, sample=False, top_k=None,
def sample_from_logits(logits, temperature=1.0, top_k=None, sample=False):
    """ Samples from logits with top_k and temperature.
    Input is of shape [batch_size, time, nb_books, softmax_size]"""

    batch = logits.shape[0]
    logits = logits[:, -1, ...]  # Take last time step.
    # Get logits at the final step, put book dimension in batch size
    logits = rearrange(logits, 'batch books emb -> (batch books) emb')
    # scale by temperature
    logits = logits / temperature
    # optionally crop probabilities to only the top k options
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    # apply softmax to convert to probabilities
    probs = F.softmax(logits, dim=-1)
    # sample from the distribution or take the most likely
    if sample:
        iz = torch.multinomial(probs, num_samples=1)
    else:
        _, iz = torch.topk(probs, k=1, dim=-1)
    iz = rearrange(iz, '(batch books) one_dim  -> batch one_dim books', batch=batch, one_dim=1)
    return iz
