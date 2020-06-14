#!/usr/bin/env python

import numpy as np
import os


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
              some target i

    Returns:
        v: a numpy array of shape (k) such that V[i] contains
           the n-dimensional V statistic for target i'''

    r = np.flip((1-cdfs), 0)
    sqr = np.square(r)
    n = r.shape[0]
    v = [1.0, r[0]]
    for i in range(1, n):
        v.append(r[i] * v[-1] - 0.5 * sqr[i-1] * v[-2])
    return v[-1]
