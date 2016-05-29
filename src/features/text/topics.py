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

NTOPICS = 100
NGRAM_RANGE = (1, 3)  # default is unigram: (1, 1)
MIN_DF = 0.01  # can be percentage or absolute value
DEBUG = False


class Topics:
    def __init__(self, column):
        self.column = column

    def fit(self, rows):
        rows = np.unique(rows.flatten())
        corpus = MyCorpus('../data/ItemInfo_train.csv', self.column, rows)

        self.bow_model = CountVectorizer(
            stop_words=stop_words, strip_accents='unicode',
            ngram_range=NGRAM_RANGE, min_df=MIN_DF)
        X = self.bow_model.fit_transform(corpus)
        if DEBUG:
            print 'vocabulary size:', len(self.bow_model.get_feature_names())

        evaluate_every = 1 if DEBUG else 0
        self.lda_model = LatentDirichletAllocation(
            NTOPICS, evaluate_every=evaluate_every)
        self.lda_model.fit(X)
        if DEBUG:
            vocab = np.asarray(self.bow_model.get_feature_names())
            for i, topic_dist in enumerate(self.lda_model.topic_word_):
                print i, vocab[np.argsort(topic_dist)[:-10:-1]]
        return self

    def transform(self, rows):
        rows, ix = np.unique(rows.flatten('F'), return_inverse=True)
        corpus = MyCorpus('../data/ItemInfo_train.csv', self.column, rows)
        X = self.bow_model.transform(corpus)
        X = self.lda_model.transform(X)
        X1 = X[ix[:(len(ix)/2)]]
        X2 = X[ix[(len(ix)/2):]]
        return np.c_[np.abs(X1 - X2), np.sum(X1 * X2, 1)]
