# -*- coding: utf-8 -*-

# Extract non-Russian words (except Russian colors)
# This was thought out mainly for the phones category.

from utils.mycorpus import MyCorpus
import numpy as np


def diff_count(rows, column, count_fns):
    rows, ix = np.unique(rows.flatten('F'), return_inverse=True)
    corpus = MyCorpus('../data/ItemInfo_train.csv', column, rows)
    # this cycle actually seems faster than list comprehension (I guess because
    # this uses numpy to store this big dataframe in memory)
    count = np.zeros((len(rows), len(count_fns)), int)
    for i, text in enumerate(corpus):
        for j, fn in enumerate(count_fns):
            count[i, j] = fn(text)
    return np.abs(count[ix[:(len(ix)/2)]] - count[ix[(len(ix)/2):]])
