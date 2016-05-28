# -*- coding: utf-8 -*-

# Extract brands: we do this by ignoring any words with cyrillic except
# for Russian color.
# Should work mostly well for such categories as mobile phones.

from utils.mycorpus import MyCorpus
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from stop_words import get_stop_words
stop_words = get_stop_words('ru')

DEBUG = False


class Topics:
    def __init__(self, column):
        self.column = column

    def fit(self, rows):
        rows = np.unique(rows.flatten())
        corpus = MyCorpus('../data/ItemInfo_train.csv', self.column, rows)

        self.bow_model = CountVectorizer(
            stop_words=stop_words, strip_accents='unicode')
        X = self.bow_model.fit_transform(corpus)
        if DEBUG:
            print 'vocabulary size:', len(self.bow_model.get_feature_names())

        self.lda_model = LatentDirichletAllocation(50)
        self.lda_model.fit(X)
        if DEBUG:
            vocab = np.asarray(self.bow_model.get_feature_names())
            for i, topic_dist in enumerate(self.model.topic_word_):
                print i, vocab[np.argsort(topic_dist)[:-10:-1]]
        return self

    def transform(self, rows):
        rows, ix = np.unique(rows.flatten('F'), return_inverse=True)
        corpus = MyCorpus('../data/ItemInfo_train.csv', self.column, rows)
        X = self.bow_model.transform(corpus)
        X = self.lda_model.transform(X)
        return X[ix[:(len(ix)/2)]] - X[ix[(len(ix)/2):]]
