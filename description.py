# -*- coding: utf-8 -*-

from tictoc import tic, toc
import csv
import sys
from gensim import corpora, models
from gensim.matutils import cossim
from stop_words import get_stop_words
stopwords = get_stop_words('ru')
import numpy as np
import pandas as pd

CATEGORY = 84

print 'load indices...'

tic()
Xpairs = np.loadtxt('data/ItemPairs_train.csv', int, delimiter=',',
                    skiprows=1, usecols=(0, 1, 2))
toc()
Xinfo = pd.read_csv('data/ItemInfo_train.csv', dtype=int, usecols=(0, 1),
                    index_col=0)
Xinfo = Xinfo[Xinfo['categoryID'] == CATEGORY]
Xinfo['row'] = np.arange(len(Xinfo))
toc()


class MyCorpus(object):
    def __iter__(self):
        with open('data/ItemInfo_train.csv', 'r') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            header = True
            for row in reader:
                if not header:
                    category = int(row[1])
                    if category == CATEGORY:
                        yield row[3].lower().split()
                header = False

print 'dictionary...'

corpus = MyCorpus()
tic()
dictionary = corpora.Dictionary(corpus, None)
toc()
dictionary.filter_extremes(2)
toc()
dictionary.filter_tokens(stopwords)
toc()
dictionary.compactify()
toc()

print 'tf-idf...'

tic()
tfidf = models.TfidfModel(dictionary=dictionary)
toc()

print 'bags...'

tic()
bags = [dictionary.doc2bow(d) for d in corpus]
toc()

print 'calculate averages...'

tic()
total = [np.sum(Xpairs[:, 2] == 0), np.sum(Xpairs[:, 2] == 1)]
avg = [0, 0]
for z, (i1, i2, dup) in enumerate(Xpairs):
    if i1 in Xinfo.index and i2 in Xinfo.index:
        sys.stdout.write('\r%d%%' % (100*z/len(Xpairs)))
        row1 = Xinfo.ix[i1]['row']
        row2 = Xinfo.ix[i2]['row']
        avg[dup] += cossim(tfidf[bags[row1]], tfidf[bags[row2]]) / total[dup]
sys.stdout.write('\r           \r')
toc()
print avg
