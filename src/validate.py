# -*- coding: utf-8 -*-

import sys
sys.dont_write_bytecode = True
import os
import pandas as pd
import numpy as np
from utils.tictoc import tic, toc

print '== load =='

tic()
info = pd.read_csv('../data/ItemInfo_train.csv',
                   dtype={'itemID': int, 'categoryID': int, 'price': float},
                   usecols=(0, 1, 6, 7, 8, 9, 10), index_col=0)
info['line'] = np.arange(len(info))
toc()
# NOTA: estou a ler apenas as primeiras N linhas
pairs = np.genfromtxt('../data/ItemPairs_train.csv', int, delimiter=',',
                      skip_header=1, usecols=(0, 1, 2), max_rows=5000)
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


def extract_categories():
    # Este encoding que eu faço aqui é por causa duma limitação do sklearn.
    # Estou a codificar categories como 83 como [0,0,0,1,0]. Ou seja, cada
    # categoria passa a ser um binário. Ele só funciona assim. Isto não é uma
    # limitação das árvores de decisão em teoria, mas é uma limitação do
    # sklearn.
    # Há outro software que podemos eventualmente usar que não precisa disto...
    tic()
    from sklearn.preprocessing import OneHotEncoder
    # NOTE: all pairs belong to the same category: we only need to use one
    categories = info.iloc[lines[:, 0]].as_matrix(['categoryID'])
    encoding = OneHotEncoder(dtype=int, sparse=False)
    categories01 = encoding.fit_transform(categories)

    df = pd.read_csv('../data/Category.csv', dtype=int, index_col=0)
    parents = df.ix[categories[:, -1]].as_matrix(['parentCategoryID'])
    encoding = OneHotEncoder(dtype=int, sparse=False)
    parents01 = encoding.fit_transform(parents)

    from utils.categorias import categorias
    names = [r'\"' + categorias[i].encode('utf8') + r'\"'
             for i in np.unique(categories)]
    names += [r'\"' + categorias[i].encode('utf8') + r'\"'
              for i in np.unique(parents)]
    toc('categories')
    return ([categories01, parents01], names)


def extract_attributes():
    X = []
    tic()
    attrbs = ('price', 'locationID', 'metroID', 'lat', 'lon')
    for attr in attrbs:
        a = info.as_matrix([attr])[:, -1]
        x = np.abs(a[lines[:, 0]] - a[lines[:, 1]])
        x[np.isnan(x)] = -10000  # NaN handling
        X.append(x)
    toc('attributes')
    return (X, attrbs)


def extract_text_counts():
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
    X = diff_count(lines, 3, count_fns)

    names = ['text-count-diff-%d' % i for i in xrange(len(count_fns))]
    names += ['text-count-both-%d' % i for i in xrange(len(count_fns))]
    toc('text counts')
    return ([X], names)


def extract_images_count():
    from features.image.imagediff import diff_image_count
    tic()
    X = diff_image_count(lines)
    toc('images count')
    return ([X], ['image-count-diff', 'image-count-both'])


def extract_brands():
    from features.text.brands import Brands
    tic()
    X1 = Brands(2).fit(lines[tr]).transform(lines)
    X2 = Brands(3).fit(lines[tr]).transform(lines)
    toc('brands')
    return ([X1, X2], ['brand-title', 'brand-descr'])


def extract_topics():
    from features.text.topics import Topics, NTOPICS
    tic()
    X = Topics(3).fit(lines[tr]).transform(lines)
    toc('topics')
    return ([X], ['topic-%d' % i for i in xrange(NTOPICS)]+['topic-dist'])


def extract_images_hash():
    if os.path.exists('../data/Images_9'):
        from features.image.imagediff import diff_image_hash
        tic()
        X = diff_image_hash(lines)
        toc('images hash')
        return ([X], ['image-hash-diff'])
    else:
        print 'Warning: images not found'
        return ([], [])

import multiprocessing
pool = multiprocessing.Pool(3)

res = [
    pool.apply_async(extract_images_hash),
    pool.apply_async(extract_topics),
    pool.apply_async(extract_text_counts),
    pool.apply_async(extract_images_count),
    pool.apply_async(extract_categories),
    pool.apply_async(extract_attributes),
    pool.apply_async(extract_brands),
]

X = []
names = []
for r in res:
    _X, _names = r.get()
    X += _X
    names += _names
for i in xrange(len(X)):  # ensure all are matrices
    if len(X[i].shape) == 1:
        X[i] = np.vstack(X[i])
X = np.concatenate(X, 1)
assert X.shape[1] == len(names)

# create model and validate

print '== model =='

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV


def kaggle_score(m, X, y):
    return roc_auc_score(y, m.predict_proba(X)[:, 1])

tic()
m = RandomForestClassifier(250)
# find a better max_depth if you can...
m = GridSearchCV(m, {'max_depth': range(20, 30+1)}, kaggle_score, n_jobs=-1)
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

if os.path.exists('/usr/bin/dot'):  # has graphviz installed?
    from sklearn.tree import DecisionTreeClassifier, export_graphviz
    m = DecisionTreeClassifier(min_samples_leaf=50)
    m.fit(X, y)
    export_graphviz(m, feature_names=names,
                    class_names=['different', 'duplicate'], label='none',
                    impurity=False, filled=True)
    os.system('dot -Tpdf tree.dot -o tree.pdf')  # compile dot file
    os.remove('tree.dot')
