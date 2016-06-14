# -*- coding: utf-8 -*-

import numpy as np
import itertools
import unicodedata
import sys

'''
most common code:
http://stackoverflow.com/questions/1518522/python-most-common-element-in-a-list
'''


def most_common(lst):
    if len(lst):
        return max(set(lst), key=lst.count)
    return None


class BaseExpressions:
    def transform(self, myreader, rows):
        column = self.get_column()
        nsymbols = self.get_nsymbols()
        ret = np.zeros((len(rows), nsymbols), int)
        rows, ix = np.unique(rows.flatten('F'), return_inverse=True)

        lenix2 = len(ix)/2
        for i, (row1, row2) in enumerate(itertools.izip(
                rows[ix[:lenix2]], rows[ix[lenix2:]])):
            if i % (lenix2/100) == 0:
                progress = (100*i)/lenix2
                sys.stdout.write('\rexpression... %2d%%' % progress)
                sys.stdout.flush()
            text1 = myreader.get_row(column, row1)
            text2 = myreader.get_row(column, row2)
            symbols1 = self.get_symbols(text1)
            symbols2 = self.get_symbols(text2)
            for j, (s1, s2) in enumerate(itertools.izip(symbols1, symbols2)):
                if s1 == s2 and s1 is not None:
                    ret[i, j] = 1
                else:
                    ret[i, j] = 0
        sys.stdout.write('\r                      \r')
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
    'Sc',
    'LC', 'Lm', 'Lt', 'Lu',
    'Pc', 'Pd', 'Pe', 'Pf',
    'Mc', 'Me', 'Mn',
    'Nl']


class UnicodeCategories(BaseExpressions):
    def get_column(self):
        return 3

    def get_nsymbols(self):
        return len(UNICODE_CATEGORIES)

    # get_most_frequent_start
    def get_symbols(self, text):
        cats = {key: list() for key in UNICODE_CATEGORIES}
        for ch in text:
            key = unicodedata.category(ch)
            if key in cats:
                cats[key].append(ch)
        return [most_common(symbs) for _, symbs in cats.iteritems()]
