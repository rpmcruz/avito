# -*- coding: utf-8 -*-

# Extract brands: we do this by ignoring any words with cyrillic except
# for Russian color.
# Should work mostly well for such categories as mobile phones.

from utils.mycorpus import MyCorpus
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
with open('features/text/colors.txt') as f:
    _colors = [unicode(line, 'utf-8').rstrip() for line in f.readlines()]

DEBUG = False


class Brands:
    def __init__(self, column):
        self.column = column

    def fit(self, filename, rows):
        rows = np.unique(rows.flatten())
        corpus = MyCorpus(filename, self.column, rows)

        import string
        from alphabet_detector import AlphabetDetector
        ad = AlphabetDetector()

        latin_vocabulary = []
        for text in corpus:
            text = text.lower()
            for ch in string.punctuation:
                text = text.replace(ch, '')
            words = [word for word in text.split()
                     if ad.is_latin(word) or word in _colors]
            latin_vocabulary += words
        latin_vocabulary = set(latin_vocabulary)
        if DEBUG:
            print latin_vocabulary

        self.tfidf_model = TfidfVectorizer(
            min_df=2, vocabulary=latin_vocabulary)
        self.tfidf_model.fit(corpus)
        return self

    def transform(self, filename, rows):
        rows, ix = np.unique(rows.flatten('F'), return_inverse=True)
        corpus = MyCorpus(filename, self.column, rows)
        X = self.tfidf_model.transform(corpus)
        ret = X[ix[:(len(ix)/2)]].multiply(X[ix[(len(ix)/2):]]).sum(1)
        if DEBUG:
            print ret
        return ret
