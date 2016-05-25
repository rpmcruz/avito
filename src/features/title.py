# -*- coding: utf-8 -*-

import preprocess
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from gensim import corpora, models
from gensim.matutils import cossim


class TitleIter:
    def __init__(self, documents):
        self.documents = documents

    def __iter__(self):
        for document in self.documents:
            words = preprocess.split_only_alphanumeric(document)
            words = preprocess.filter_russian_except_colors(words)
            yield words


class ExtractTitle(BaseEstimator, ClassifierMixin):
    def __init__(self, column):
        self.classes_ = (0, 1)
        self.column = column

    def fit(self, X):
        rows = np.unique(X.flatten())
        corpus = preprocess.Documents('../data/ItemInfo_train.csv',
                                      self.column, rows)
        self.dictionary = corpora.Dictionary(TitleIter(corpus), None)
        self.dictionary.filter_extremes(3)
        self.dictionary.compactify()
        self.tfidf = models.TfidfModel(dictionary=self.dictionary)
        return self

    def transform(self, X):
        rows, ix = np.unique(X.flatten(), return_inverse=True)
        assert len(ix) % 2 == 0  # must be even
        corpus = preprocess.Documents('../data/ItemInfo_train.csv',
                                      self.column, rows)
        corpus = TitleIter(corpus)
        bags = [self.dictionary.doc2bow(d) for d in corpus]
        ret = []
        for i in xrange(0, len(ix), 2):
            b1 = bags[ix[i]]
            b2 = bags[ix[i+1]]
            dist = cossim(self.tfidf[b1], self.tfidf[b2])
            ret.append(dist)
        return ret
