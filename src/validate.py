# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from utils.tictoc import tic, toc

CATEGORY = 84  # 0 to disable

# load some basic stuff
print 'load...'

tic()
Xinfo = pd.read_csv('../data/ItemInfo_train.csv',
                    dtype={'itemID': int, 'categoryID': int, 'price': float},
                    usecols=(0, 1, 6), index_col=0)
images = []
for arr in Xinfo['images_array']:
    if np.isnan(arr):
        images.append(None)
    else:
        img = arr.split(', ')[0]
        dirname = 'Images_%s/%s' % (img[-2], img[-1])
        filename = str(int(img)) + '.jpg'
        images.append((dirname, filename))
Xinfo['line'] = np.arange(len(Xinfo))
toc()
Xpairs = np.loadtxt('../data/ItemPairs_train.csv', int, delimiter=',',
                    skiprows=1, usecols=(0, 1, 2))
if CATEGORY:
    Xinfo = Xinfo[Xinfo['categoryID'] == CATEGORY]
    Xpairs = [(i1, i2, d) for i1, i2, d in Xpairs
              if i1 in Xinfo.index and i2 in Xinfo.index]
toc()

# cross-validation
print 'cross-validation...'

tic()
X = np.asarray(
    [(Xinfo.ix[i1]['line'], Xinfo.ix[i2]['line']) for i1, i2, d in Xpairs])
y = np.asarray([d for i1, i2, d in Xpairs])
toc()

print '%%dup: %.2f' % (np.sum(y == 1)/float(len(y)))

#from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from features.title import ExtractTitle


def evaluate(args):
    tr, ts = args
    tic()
    X1 = ExtractTitle(2).fit(X[tr], y).transform(X[ts], y[ts])
    toc()
    X2 = ExtractTitle(3).fit(X[tr], y).transform(X[ts], y[ts])
    toc()
    X3 = np.abs(Xinfo.iloc[X[tr][:, 0]]['price'] -
                Xinfo.iloc[X[tr][:, 1]]['price'])
    toc()
    # X4 = lista de hashes
    #toc()
    _X = np.c_[X1, X2, X3]

    tic()
    m = RandomForestClassifier(100, max_depth=8)
    m.fit(_X[tr], y[tr])
    yp = m.predict(_X[ts])
    toc()

    print 'baseline: %.4f - title: %.4f (FP: %.2f FN: %.2f)' % (
        np.sum(y[ts] == 0)/float(len(ts)), accuracy_score(y[ts], yp),
        np.sum(np.logical_and(yp == 1, y[ts] == 0))/float(np.sum(y[ts] == 0)),
        np.sum(np.logical_and(yp == 0, y[ts] == 1))/float(np.sum(y[ts] == 1)))

"""
import multiprocessing
p = multiprocessing.Pool(4)
folds = StratifiedKFold(y)
p.map(evaluate, folds)
"""
evaluate((np.arange(len(y)), np.arange(len(y))))
