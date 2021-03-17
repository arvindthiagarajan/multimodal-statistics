#!/usr/bin/env python

import numpy as np
import os

from collections import Counter
from scipy.special import factorial
from scipy.stats.mstats import gmean


def get_rank_cdf(ranks, n=None, default_weight=1.0):

    """Computes the cumulative distribution function (CDF) given a list of
       sampled ranks.

    Args:
        ranks:          a list of rank elements, where each element is either
                        an integral rank or a tuple (rank, weight). the lowest
                        (best) possible rank is 0
        n:              (optional) the highest (worst) possible rank.
                        inferred from `ranks` if not provided
        default_weight: (optional) float giving the probability to be assigned
                        to any sample for which a weight is not provided

    Returns:
        cdf: a numpy array of shape (n) such that cdf[i] is the probability
             that a rank is less (better) than i"""

    ranks = [
        rank if isinstance(rank, tuple) else (rank, default_weight) for rank in ranks
    ]
    if n is None:
        n = max([rank[0] for rank in ranks])
    pdf = np.zeros(n)
    for rank in ranks:
        pdf[rank[0]] += rank[1]
    cdf = np.cumsum(pdf)
    return cdf / cdf[-1]


def stuart_strength(cdfs):

    """Computes the n-dimensional V statistic (Q-value / n!) given a set of cdfs

    Args:
        cdfs: a numpy array of shape (n,k) such that cdfs[:,i] corresponds
              to the output of a call to get_rank_cdf with the ranks for
              some element i

    Returns:
        v: a numpy array of shape (k) such that V[i] contains
           the n-dimensional V statistic for element i"""

    r = np.flip((1 - cdfs), 0)
    sqr = np.square(r)
    n = r.shape[0]
    v = [1.0, r[0]]
    for i in range(1, n):
        v.append(r[i] * v[-1] - 0.5 * sqr[i - 1] * v[-2])
    return v[-1]


def fisher_p_value(inputs, weights):

    """Computes a combined p-value using a novel weighted
       variant of Fisher's method

    Args:
        inputs: a numpy array of shape (n,k) such that inputs[:, i] corresponds
                to the p-values for null hypothesis i from all n different
                experiments
        weights: a numpy array of shape (n,) such that weights[j] is the
                 confidence weight assigned to experiment j

    Returns:
        a numpy array of shape (k,) giving the combined p-values for each
        of k null hypotheses"""

    mask = weights > 0
    inputs, weights = inputs[mask], weights[mask]

    nlogps = -np.log(inputs)
    counts = Counter(weights)

    if len(counts) == 1:
        f = 1
        v = nlogps.sum(0)
        for i in range(nlogps.shape[0] - 1, 0, -1):
            f = f * v / i + 1
        return (v - np.log(f)).squeeze()

    weights = weights / gmean(weights)
    v = np.dot(weights, nlogps)
    counts = [(0, 1)] + [(1 / w, count) for w, count in counts.items()]
    counts = sorted(counts, key=lambda x: -x[1])
    a, n = map(np.array, zip(*counts))
    batch_sizes = enumerate(np.diff(np.flip(n - 1), prepend=0))
    batch_sizes = sum([[n.shape[0] - i] * c for i, c in batch_sizes], [])

    adiff = a[:, None] - a[None, :]
    np.fill_diagonal(adiff, 1)
    adiff = 1 / adiff
    c = [np.prod(np.power(adiff, -n), axis=1).flatten()]
    np.fill_diagonal(adiff, 0)

    amats = []
    factor = np.exp(-v[:, None] * a[None, :])
    factor *= np.power(v[:, None], n[None, :] - 1)
    factor /= factorial(n - 1)
    lhpval = np.dot(factor, c[0])
    for i, size in enumerate(batch_sizes):
        factor *= n - i - 1
        factor /= v[:, None]
        amats.append(
            adiff[:size] if len(amats) == 0 else amats[-1][:size] * adiff[:size]
        )
        terms = [prevc[:size] * np.dot(amat[:size], n) for prevc, amat in zip(c, amats)]
        c.insert(0, np.array(terms).mean(axis=0))
        lhpval += np.dot(factor[:, :size], c[0])

    return 1 - lhpval
