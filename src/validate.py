# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from utils.tictoc import tic, toc

print '== load =='

tic()
info = pd.read_csv('../data/ItemInfo_train.csv',
                   dtype={'itemID': int, 'categoryID': str, 'price': float},
                   usecols=(0, 1, 6, 7, 8, 9, 10), index_col=0)
info['line'] = np.arange(len(info))
toc()

# NOTA: estou a ler apenas as primeiras 1000 linhas
pairs = np.genfromtxt('../data/ItemPairs_train.csv', int, delimiter=',',
                      skip_header=1, usecols=(0, 1, 2), max_rows=2500)
toc()

# transforma ItemID em linhas do ficheiro CSV e da matriz info
tic()
lines = np.asarray(
    [(info.ix[i1]['line'], info.ix[i2]['line']) for i1, i2, d in pairs], int)
y = np.asarray([d for i1, i2, d in pairs], int)
toc()

print '== extract features =='

idx = np.arange(len(lines))  # split train and test
np.random.shuffle(idx)
tr = idx[:int(0.60*len(lines))]
ts = idx[int(0.60*len(lines)):]

X = []  # features

print 'extract categories'

# Este encoding que eu faço aqui é por causa duma limitação do sklearn.
# Estou a codificar categories como 83 como [0,0,0,1,0]. Ou seja, cada
# categoria passa a ser um binário. Ele só funciona assim. Isto não é uma
# limitação das árvores de decisão em teoria, mas é uma limitação do sklearn.
# Há outro software que podemos eventualmente usar que não precisa disto...
tic()
from sklearn.preprocessing import OneHotEncoder
categories = info.as_matrix(['categoryID'])
encoding = OneHotEncoder(dtype=int, sparse=False)
categories = encoding.fit_transform(info.as_matrix(['categoryID']))
# NOTE: all pairs belong to the same category: we only need to use one
X.append(categories[lines[:, 0]])
toc()

print 'extract other attributes'

tic()
for attr in ('price', 'locationID', 'metroID', 'lat', 'lon'):
    a = info.as_matrix([attr])[:, -1]
    x = np.abs(a[lines[:, 0]] - a[lines[:, 1]])
    x[np.isnan(x)] = -10000  # NaN handling
    X.append(x)
toc()

print 'extract text count differences'

from features.text.count import diff_count
count_fns = [
    lambda text: text.count(','),
    lambda text: text.count('.'),
    lambda text: text.count('!'),
    lambda text: text.count('-'),
    lambda text: text.count('*'),
    lambda text: text.count(u'•'),
    lambda text: len(text),
]
X.append(diff_count(lines, 3, count_fns))
toc()

print 'extract brands as distance'

from features.text.brands import Brands
tic()
X.append(Brands(2).fit(lines[tr]).transform(lines))
toc()

print 'extract topics differences'

from features.text.topics import Topics
tic()
X.append(Topics(3).fit(lines[tr]).transform(lines))
toc()

print 'extract images hash differences'

if os.path.exists('../data/Images_9'):
    from features.image.imagediff import diff_image_hash
    tic()
    X.append(diff_image_hash(lines))
    toc()
else:
    print 'Warning: images not found'

# create model and validate

print '== model =='

for i in xrange(len(X)):  # ensure all entries are matrices
    if len(X[i].shape) == 1:
        X[i] = np.vstack(X[1])
X = np.concatenate(X, 1)

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV

tic()
m = RandomForestClassifier(100, max_depth=14)
# find a better max_depth if you can...
m = GridSearchCV(m, {'max_depth': range(10, 24+1)}, n_jobs=-1)
m.fit(X[tr], y[tr])
toc()
pp = m.predict_proba(X[ts])[:, 1]
yp = pp >= 0.5
toc()

print 'best params:', m.best_params_
print

print 'baseline: %.4f' % (np.sum(y[ts] == 0)/float(len(ts)))
print 'y=0 | TN=%.2f | FP=%.2f |\ny=1 | FN=%.2f | TP=%.2f |' % (1, 0, 1, 0)
print

print 'our model: %.4f' % accuracy_score(y[ts], yp)
(TN, FP), (FN, TP) = confusion_matrix(y[ts], yp)
print 'y=0 | TN=%.2f | FP=%.2f |\ny=1 | FN=%.2f | TP=%.2f |' % (
    TN / float(np.sum(y[ts] == 0)), FP / float(np.sum(y[ts] == 0)),
    FN / float(np.sum(y[ts] == 1)), TP / float(np.sum(y[ts] == 1)))

print
print 'kaggle score:', roc_auc_score(y[ts], pp)

DRAW_TREE = False  # this does not work in Windows
if DRAW_TREE:
    import os
    from sklearn.tree import DecisionTreeClassifier, export_graphviz
    m = DecisionTreeClassifier(max_depth=6)
    m.fit(X, y)
    export_graphviz(m,  # feature_names=['title', 'description', 'dprice'],
                    class_names=['different', 'duplicate'], label='none',
                    impurity=False, filled=True)
    os.system('dot -Tpdf tree.dot -o ../tree.pdf')  # compile dot file
    os.remove('tree.dot')
