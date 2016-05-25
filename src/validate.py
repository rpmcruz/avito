# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from utils.tictoc import tic, toc

print 'load...'

tic()
info = pd.read_csv('../data/ItemInfo_train.csv',
                   dtype={'itemID': int, 'categoryID': str, 'price': float},
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

"""
images = []
for arr in Xinfo['images_array']:
    if np.isnan(arr):
        images.append(None)
    else:
        img = arr.split(', ')[0]
        dirname = 'Images_%s/%s' % (img[-2], img[-1])
        filename = str(int(img)) + '.jpg'
        images.append((dirname, filename))
"""

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from features.title import ExtractTitle

prices = info.as_matrix(['price'])[:, -1]
prices[np.isnan(prices)] = -10000  # HACK: some prices are NaN

# Este encoding que eu faço aqui é por causa duma limitação do sklearn
# Estou a codificar categories como 83 como [0,0,0,1,0]. Ou seja, cada
# categoria passa a ser um binário. Ele só funciona assim. Isto não é uma
# limitação das árvores de decisão em teoria, mas é uma limitação do sklearn.
# Há outro software que podemos eventualmente usar que não precisa disto...
categories = info.as_matrix(['categoryID'])
encoding = OneHotEncoder(dtype=int, sparse=False)
categories = encoding.fit_transform(info.as_matrix(['categoryID']))

idx = np.arange(len(lines))
np.random.shuffle(idx)
tr = idx[:(0.70*len(lines))]
ts = idx[(0.70*len(lines)):]

print 'extract features...'

tic()
X1 = ExtractTitle(2).fit(lines).transform(lines)
toc()
X2 = ExtractTitle(3).fit(lines).transform(lines)
toc()
X3 = np.abs(prices[lines[:, 0]] - prices[lines[:, 1]])
toc()
X4 = categories[lines[:, 0]]  # does not matter: they are the same
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

DRAW_TREE = False  # this does not work in Windows
if DRAW_TREE:
    import os
    from sklearn.tree import DecisionTreeClassifier, export_graphviz
    m = DecisionTreeClassifier(max_depth=4)
    m.fit(X, y)
    export_graphviz(m,  # feature_names=['title', 'description', 'dprice'],
                    class_names=['different', 'duplicate'], label='none',
                    impurity=False, filled=True)
    os.system('dot -Tpdf tree.dot -o ../tree.pdf')  # compile dot file
    os.remove('tree.dot')
