#!/usr/bin/env python

import numpy as np
import os

from scipy.special import logsumexp


def get_rank_cdf(ranks, n=None, default_weight=1.0):

    '''Computes the cumulative distribution function (CDF) given a list of
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
             that a rank is less (better) than i'''

    ranks = [rank if isinstance(rank, tuple) else (rank, default_weight)
             for rank in ranks]
    if n is None:
        n = max([rank[0] for rank in ranks])
    pdf = np.zeros(n)
    for rank in ranks:
        pdf[rank[0]] += rank[1]
    cdf = np.cumsum(pdf)
    return cdf/cdf[-1]


def stuart_strength(cdfs):

    '''Computes the n-dimensional V statistic (Q-value / n!) given a set of cdfs

    Args:
        cdfs: a numpy array of shape (n,k) such that cdfs[:,i] corresponds
              to the output of a call to get_rank_cdf with the ranks for
              some element i

    Returns:
        v: a numpy array of shape (k) such that V[i] contains
           the n-dimensional V statistic for element i'''

    r = np.flip((1-cdfs), 0)
    sqr = np.square(r)
    n = r.shape[0]
    v = [1.0, r[0]]
    for i in range(1, n):
        v.append(r[i] * v[-1] - 0.5 * sqr[i-1] * v[-2])
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
        a numpy array of shape (k,) giving the negative log of the combined
        p-values for each of k null hypotheses"""

    unique_weights = np.unique(weights).tolist()
    wt = np.copy(weights)
    nlogps = -np.log(inputs)

    # Combine p-values with identical weights iteratively
    # until all effective weights are distinct

    while len(unique_weights) != wt.shape[0]:
        new_logps = []
        new_wts = []
        for w in unique_weights:
            s = nlogps[wt == w]
            v = s.sum(0)
            f = 1
            for i in range(s.shape[0] - 1, 0, -1):
                f = f * v / i + 1
            new_logps.append(v - np.log(f))
            new_wts.append(s.shape[0] * w)
        nlogps = np.array(new_logps)
        wt = np.array(new_wts)
        unique_weights = np.unique(wt).tolist()

    if len(unique_weights) == 1:
        return nlogps.squeeze()

    # Combine p-values with distinct weights

    stat = np.dot(wt, nlogps)
    stat = (wt.shape[0] - 1) * np.log(wt) - stat[:, None] / wt[None, :]

    wt = wt[:, None] - wt[None, :]
    wt[wt == 0] = 1
    wt = np.prod(wt, axis=1).flatten()
    return -logsumexp(stat, axis=1, b=1 / wt)
