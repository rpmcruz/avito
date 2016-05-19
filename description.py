# -*- coding: utf-8 -*-

from tictoc import tic, toc
import csv
from gensim import corpora, models
from gensim.matutils import cossim
from stop_words import get_stop_words
stopwords = get_stop_words('ru')

CATEGORY = 84


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
bags = (dictionary.doc2bow(d) for d in corpus)
toc()
tfidf = models.TfidfModel(dictionary=dictionary)
toc()

for v in bags:
    for u in bags:
        print cossim(tfidf[u], tfidf[v])
