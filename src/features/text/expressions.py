# -*- coding: utf-8 -*-

# Returns whether both texts have some expression in common.

from utils.mycorpus import MyCorpus
import numpy as np

_symbols = ['*', '-', '+', '=', u'•', '1)', 'a)']


class Expressions:
    def __init__(self, column):
        self.column = column

    def transform(self, filename, rows, column):
        rows, ix = np.unique(rows.flatten('F'), return_inverse=True)
        corpus = MyCorpus(filename, column, rows)
        # this cycle actually seems faster than list comprehension (I guess because
        # this uses numpy to store this big dataframe in memory)
        has = np.zeros((len(rows), len(count_fns)), int)
        for i, text in enumerate(corpus):
            for j, fn in enumerate(count_fns):
                has[i, j] = self.has_expression(text)
        has1 = has[ix[:(len(ix)/2)]]
        has2 = has[ix[(len(ix)/2):]]
        return ((has1 + has2) == 2).astype(int)



class Enumerations(Expressions):
    def has_expression(self, text):
        lines = text.split('\n')

        for symbol in _symbols:
            has = False
            for line in lines:
                _has = line.startswith(

