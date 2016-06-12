# -*- coding: utf-8 -*-

import numpy as np
import itertools
import unicodedata

'''
most common code:
http://stackoverflow.com/questions/1518522/python-most-common-element-in-a-list
'''


def most_common(lst):
    return max(set(lst), key=lst.count)


class BaseExpressions:
    def transform(self, myreader, rows):
        column = self.get_column()
        nsymbols = self.get_nsymbols()
        ret = np.zeros((len(rows), nsymbols), int)
        rows, ix = np.unique(rows.flatten('F'), return_inverse=True)

        for i, (row1, row2) in enumerate(itertools.izip(
                rows[ix[:(len(ix)/2)]], rows[ix[(len(ix)/2):]])):
            text1 = myreader.get_row(column, row1)
            text2 = myreader.get_row(column, row2)
            symbols1 = set(self.get_symbols(text1))
            symbols2 = set(self.get_symbols(text2))
            for j, (s1, s2) in enumerate(itertools.izip(symbols1, symbols2)):
                if s1 == s2:
                    ret[i, j] = 1
                else:
                    ret[i, j] = 0
        return ret


class StartsWith(BaseExpressions):
    def get_column(self):
        return 3

    def get_nsymbols(self):
        return 1

    # get_most_frequent_start
    def get_symbols(self, text):
        lines = text.split('\n')
        symbols = [line[:2] for line in lines]
        return [most_common(symbols)]


UNICODE_CATEGORIES = [
    'Sc', 'Sk', 'Sm', 'So',
    'LC', 'Ll', 'Lm', 'Lo', 'Lt', 'Lu',
    'Pc', 'Pd', 'Pe', 'Pf', 'Pi', 'Po', 'Ps',
    'Mc', 'Me', 'Mn',
    'Nl', 'No']


class UnicodeCategories(BaseExpressions):
    def get_column(self):
        return 3

    def get_nsymbols(self):
        return len(UNICODE_CATEGORIES)

    # get_most_frequent_start
    def get_symbols(self, text):
        cats = dict.fromkeys(UNICODE_CATEGORIES, [])
        for ch in text:
            key = unicodedata.category(ch)
            if key in cats:
                cats[key].append(ch)
        return [most_common(symbs) for _, symbs in cats.iteritems()]
