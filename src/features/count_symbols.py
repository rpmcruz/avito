# -*- coding: utf-8 -*-

# Extract non-Russian words (except Russian colors)
# This was thought out mainly for the phones category.

import preprocess
import numpy as np


def diff_count_symbols(rows, column, symbols):
    rows, ix = np.unique(rows.flatten('F'), return_inverse=True)
    corpus = preprocess.Documents('../data/ItemInfo_train.csv', column, rows)
    # this cycle actually seems faster than list comprehension (I guess because
    # this uses numpy to store this big dataframe in memory)
    count = np.zeros((len(rows), len(symbols)), int)
    for i, text in enumerate(corpus):
        for j, symbol in enumerate(symbols):
            count[i, j] = text.count(symbol)
    return np.abs(count[ix[:(len(ix)/2)]] - count[ix[(len(ix)/2):]])


def diff_length(rows, column):
    rows, ix = np.unique(rows.flatten('F'), return_inverse=True)
    corpus = preprocess.Documents('../data/ItemInfo_train.csv', column, rows)
    # this cycle actually seems faster than list comprehension (I guess because
    # this uses numpy to store this big dataframe in memory)
    count = np.zeros(len(rows), int)
    for i, text in enumerate(corpus):
        count[i] = len(text)
    return np.abs(count[ix[:(len(ix)/2)]] - count[ix[(len(ix)/2):]])
