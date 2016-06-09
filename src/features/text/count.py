# -*- coding: utf-8 -*-

# Returns matrix of differences between certain count functions, as well as
# a column saying whether any of those counts is present in both articles too.

from utils.mycorpus import MyCorpus
import numpy as np


def _counts(filename, rows, column, count_fns):
    rows, ix = np.unique(rows.flatten('F'), return_inverse=True)
    corpus = MyCorpus(filename, column, rows)
    # this cycle actually seems faster than list comprehension (I guess because
    # this uses numpy to store this big dataframe in memory)
    count = np.zeros((len(rows), len(count_fns)), int)
    for i, text in enumerate(corpus):
        for j, fn in enumerate(count_fns):
            count[i, j] = fn(text)
    c1 = count[ix[:(len(ix)/2)]]
    c2 = count[ix[(len(ix)/2):]]
    return (c1, c2)


def diff_count(filename, rows, column, count_fns):
    c1, c2 = _counts(filename, rows, column, count_fns)
    return np.abs(c1 - c2)


def both_count(filename, rows, column, count_fns):
    c1, c2 = _counts(filename, rows, column, count_fns)
    return np.logical_and(c1 > 0, c2 > 0)
