# -*- coding: utf-8 -*-

# Extract brands: we do this by ignoring any words with cyrillic except
# for Russian color.
# Should work mostly well for such categories as mobile phones.

import numpy as np
import itertools
import string

# colors in Russian
color_filename = 'features/text/colors.txt'
if __name__ == '__main__':
    color_filename = 'colors.txt'
with open(color_filename) as f:
    _colors = [unicode(line, 'utf-8').rstrip() for line in f.readlines()]

# package to guess to see if word is latin or cyrillic
from alphabet_detector import AlphabetDetector
ad = AlphabetDetector()


class Terms:
    def __init__(self, column):
        self.column = column

    def get_words(self, text):
        text = text.lower()
        for ch in string.punctuation:
            text = text.replace(ch, '')
        return [word for word in text.split() if self.condition(word)]

    def transform(self, myreader, rows):
        # very odd: this being dtype=int or float does not seem to matter
        # in other words, only same or different seem to matter
        ret = np.zeros(len(rows))
        rows, ix = np.unique(rows.flatten('F'), return_inverse=True)

        for i, (row1, row2) in enumerate(itertools.izip(
                rows[ix[:(len(ix)/2)]], rows[ix[(len(ix)/2):]])):
            text1 = myreader.get_row(self.column, row1)
            text2 = myreader.get_row(self.column, row2)
            words1 = set(self.get_words(text1))
            words2 = set(self.get_words(text2))
            common = words1 & words2
            den = min(len(words1), len(words2))
            if den > 0:
                ret[i] = len(common) / float(den)
            else:
                ret[i] = 0
        return ret


class Brands(Terms):
    def condition(self, word):
        return ad.is_latin(word) or word in _colors


class Topics(Terms):
    def condition(self, word):
        return not ad.is_latin(word)


if __name__ == '__main__':
    rows = np.asarray([[0, 1], [0, 2], [0, 0], [0, 5]])
    print Brands(2).transform('../../../data/ItemInfo_train.csv', rows)
