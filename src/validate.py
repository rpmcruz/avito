# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from utils.tictoc import tic, toc

print 'load...'

tic()
info = pd.read_csv('../data/ItemInfo_train.csv',
                   dtype={'itemID': int, 'categoryID': int, 'price': float},
                   usecols=(0, 1, 4, 6), index_col=0)
info['line'] = np.arange(len(info))
toc()

# NOTA: estou a ler apenas as primeiras 1000 linhas
Xpairs = np.genfromtxt('../data/ItemPairs_train.csv', int, delimiter=',',
                       skip_header=1, usecols=(0, 1, 2), max_rows=1000)
toc()

# transforma ItemID em linhas do ficheiro CSV e da matriz info
tic()
lines = np.asarray(
    [(info.ix[i1]['line'], info.ix[i2]['line']) for i1, i2, d in Xpairs], int)
y = np.asarray([d for i1, i2, d in Xpairs], int)
toc()

print 'cross-validation...'

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from features.title import ExtractTitle

Xinfo = info.as_matrix(['price', 'categoryID'])

# HACK: some prices are NaN
Xinfo[:, 0][np.isnan(Xinfo[:, 0])] = -10000

idx = np.arange(len(lines))
np.random.shuffle(idx)
tr = idx[:(0.70*len(lines))]
ts = idx[(0.70*len(lines)):]

tic()
X1 = ExtractTitle(2).fit(lines).transform(lines)
toc()
X2 = ExtractTitle(3).fit(lines).transform(lines)
toc()
X3 = np.abs(Xinfo[lines[:, 0], 0] - Xinfo[lines[:, 1], 0])
toc()
X4 = Xinfo[lines[:, 0], 1] == Xinfo[lines[:, 1], 1]
toc()
# X5 = lista de hashes
#toc()
X = np.c_[X1, X2, X3, X4]

tic()
m = RandomForestClassifier(100, max_depth=8)
m.fit(X[tr], y[tr])
yp = m.predict(X[ts])
toc()

print 'baseline: %.4f' % (np.sum(y[ts] == 0)/float(len(ts)))
print 'y=0 | TN=%.2f | FP=%.2f |\ny=1 | FN=%.2f | TP=%.2f |' % (1, 0, 1, 0)
print

print 'our model: %.4f' % accuracy_score(y[ts], yp)
(TN, FP), (FN, TP) = confusion_matrix(y[ts], yp)
print 'y=0 | TN=%.2f | FP=%.2f |\ny=1 | FN=%.2f | TP=%.2f |' % (
    TN / float(np.sum(y[ts] == 0)), FP / float(np.sum(y[ts] == 0)),
    FN / float(np.sum(y[ts] == 1)), TP / float(np.sum(y[ts] == 1)))
