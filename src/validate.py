# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from utils.tictoc import tic, toc

CATEGORY = 84  # 0 to disable

# load some basic stuff
print 'load...'

tic()
Xinfo = pd.read_csv('../data/ItemInfo_train.csv', dtype=int, usecols=(0, 1),
                    index_col=0)
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

from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score
from features.title import ExtractTitle

for i, (tr, ts) in enumerate(StratifiedKFold(y)):
    tic()
    m = ExtractTitle().fit(X[tr], y)
    #print m.dictionary.token2id  # DEBUG
    yp = m.transform(X[ts], y[ts])
    # TODO:
    # - add other features
    # - use something like a forest
    print 'fold %d - baseline: %.4f - title: %.4f' % (
        i+1, np.sum(y[ts] == 0)/float(len(ts)), accuracy_score(y[ts], yp))
    toc()
