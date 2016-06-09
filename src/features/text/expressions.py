# -*- coding: utf-8 -*-

import numpy as np
import itertools

'''
most common code:
http://stackoverflow.com/questions/1518522/python-most-common-element-in-a-list
'''


def most_common(lst):
    return max(set(lst), key=lst.count)


class StartsWith:
    def __init__(self, column):
        self.column = column

    def get_most_frequent_start(self, text):
        lines = text.split('\n')
        symbols = [line[:2] for line in lines]
        return most_common(symbols)

    def transform(self, myreader, rows):
        ret = np.zeros(len(rows), int)
        rows, ix = np.unique(rows.flatten('F'), return_inverse=True)

        for i, (row1, row2) in enumerate(itertools.izip(
                rows[ix[:(len(ix)/2)]], rows[ix[(len(ix)/2):]])):
            text1 = myreader.get_row(self.column, row1)
            text2 = myreader.get_row(self.column, row2)
            symbol1 = set(self.get_most_frequent_start(text1))
            symbol2 = set(self.get_most_frequent_start(text2))
            if symbol1 == symbol2:
                ret[i] = 1
            else:
                ret[i] = 0
        return ret
