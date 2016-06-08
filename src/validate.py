#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sys
sys.dont_write_bytecode = True
import os
import pandas as pd
import numpy as np
from utils.mycorpus import MyCSVReader
from utils.tictoc import tic, toc

N = 1000
#np.random.seed(124)

FINAL_SUBMISSION = False

print '== load =='

tic()
filename_tr = '../data/ItemInfo_train.csv'
info_tr = pd.read_csv(filename_tr,
                      dtype={'itemID': int, 'categoryID': int, 'price': float},
                      usecols=(0, 1, 6, 7, 8, 9, 10), index_col=0)
info_tr['line'] = np.arange(len(info_tr))
if FINAL_SUBMISSION:
    filename_ts = '../data/ItemInfo_test.csv'
    info_ts = pd.read_csv(filename_ts,
                          dtype={'itemID': int, 'categoryID': int,
                                 'price': float},
                          usecols=(0, 1, 6, 7, 8, 9, 10), index_col=0)
    info_ts['line'] = np.arange(len(info_ts))
else:
    filename_ts = filename_tr
    info_ts = info_tr
toc('item info')

myreader_tr = MyCSVReader(filename_tr)
if filename_tr == filename_ts:
    myreader_ts = myreader_tr
else:
    myreader_ts = MyCSVReader(filename_ts)
toc('lines to seek')

# NOTA: estou a ler apenas as primeiras N linhas
pairs_tr = np.genfromtxt('../data/ItemPairs_train.csv', int, delimiter=',',
                         skip_header=1, usecols=(0, 1, 2))
if FINAL_SUBMISSION:
    pairs_ts = np.genfromtxt('../data/ItemPairs_test.csv', int, delimiter=',',
                             skip_header=1, usecols=(1, 2))
    ytr = pairs_tr[:, -1]
else:
    pairs_tr = pairs_tr[  # undersample to speedup things
        np.random.choice(np.arange(len(pairs_tr)), N, False)]
    # split train into train and test
    idx = np.arange(len(pairs_tr))
    np.random.shuffle(idx)
    tr = idx[:int(0.40*len(pairs_tr))]
    ts = idx[int(0.40*len(pairs_tr)):]
    pairs_ts = pairs_tr[ts]
    pairs_tr = pairs_tr[tr]
    ytr = pairs_tr[:, -1]
    yts = pairs_ts[:, -1]
    pairs_ts = pairs_ts[:, :-1]  # drop dups
toc('pairs')

# transforma ItemID em linhas do ficheiro CSV e da matriz info
lines_tr = np.asarray(
    [(info_tr.ix[i1]['line'], info_tr.ix[i2]['line'])
     for i1, i2, d in pairs_tr], int)
lines_ts = np.asarray(
    [(info_ts.ix[i1]['line'], info_ts.ix[i2]['line'])
     for i1, i2 in pairs_ts], int)
toc('pairs to lines')

print '== extract features =='

# Each extraction method must return a tuple with: (list of features from
# training, list of features from test, name of the features)


def extract_categories():
    tic()
    # Este encoding que eu faço aqui é por causa duma limitação do sklearn.
    # Estou a codificar categories como 83 como [0,0,0,1,0]. Ou seja, cada
    # categoria passa a ser um binário. Ele só funciona assim. Isto não é uma
    # limitação das árvores de decisão em teoria, mas é uma limitação do
    # sklearn.
    # Há outro software que podemos eventualmente usar que não precisa disto...
    # O xgboost tb não suporta categóricas.

    from sklearn.preprocessing import OneHotEncoder
    # NOTE: all pairs belong to the same category: we only need to use one
    encoding = OneHotEncoder(dtype=int, sparse=False)
    categories_tr = info_tr.iloc[lines_tr[:, 0]].as_matrix(['categoryID'])
    categories_ts = info_ts.iloc[lines_ts[:, 0]].as_matrix(['categoryID'])
    all_categories = np.r_[categories_tr, categories_ts]
    encoding.fit(all_categories)
    categories01_tr = encoding.transform(categories_tr)
    categories01_ts = encoding.transform(categories_ts)

    df = pd.read_csv('../data/Category.csv', dtype=int, index_col=0)
    encoding = OneHotEncoder(dtype=int, sparse=False)
    parents_tr = df.ix[categories_tr[:, -1]].as_matrix(['parentCategoryID'])
    parents_ts = df.ix[categories_ts[:, -1]].as_matrix(['parentCategoryID'])
    all_parents = np.r_[parents_tr, parents_ts]
    encoding.fit(all_parents)
    parents01_tr = encoding.transform(parents_tr)
    parents01_ts = encoding.transform(parents_ts)

    from utils.categorias import categorias
    names = [categorias[i] for i in np.unique(all_categories)]
    names += [categorias[i] for i in np.unique(all_parents)]
    toc('categories')
    return ([categories01_tr, parents01_tr], [categories01_ts, parents01_ts],
            names)


def extract_attributes():
    tic()
    Xtr = []
    Xts = []
    # not using 'locationID' because it degrades performance
    attrbs = ['price', 'metroID']
    for X, info, lines in ((Xtr, info_tr, lines_tr), (Xts, info_ts, lines_ts)):
        for attr in attrbs:
            a = info.as_matrix([attr])[:, -1]
            x = np.abs(a[lines[:, 0]] - a[lines[:, 1]])
            x[np.isnan(x)] = 10000  # NaN handling
            X.append(x)
        # lat, lon use euler distance
        # using lat,lon individually degrades performance, but this metric
        # seems to improve it slightly
        l1 = info.as_matrix(['lon'])[:, -1]
        l2 = info.as_matrix(['lat'])[:, -1]
        x = (l1[lines[:, 0]] - l2[lines[:, 1]]) ** 2
        x[np.isnan(x)] = 10000  # NaN handling
        X.append(x)
    toc('attributes')
    return (Xtr, Xts, attrbs + ['lon-lat'])


def extract_text_expressions():
    tic()
    _myreader_tr = myreader_tr.copy()
    _myreader_ts = myreader_ts.copy()

    from features.text.expressions import StartsWith
    Xtr = StartsWith(3).transform(_myreader_tr, lines_tr)
    Xts = StartsWith(3).transform(_myreader_ts, lines_ts)

    names = ['common-start']
    toc('text expressions')
    return ([Xtr], [Xts], names)


def extract_text_counts():
    tic()
    from features.text.count import diff_count, both_count
    # symbols tested that were not useful: +, *, 1), a)
    count_fns = [
        lambda text: text.count('.'),  # 1
        lambda text: text.count('!'),  # 2
        lambda text: text.count('_'),  # 6
        lambda text: text.count('='),  # 9
        lambda text: text.count(u'•'),  # 10
        lambda text: len(text),  # 11
    ]
    Xtr1 = diff_count(filename_tr, lines_tr, 3, count_fns)
    Xts1 = diff_count(filename_ts, lines_ts, 3, count_fns)
    names = ['text-count-diff-%d' % i for i in xrange(len(count_fns))]

    count_fns = [
        lambda text: text.count(','),  # 0
        lambda text: text.count('-'),  # 3
    ]
    Xtr2 = both_count(filename_tr, lines_tr, 3, count_fns)
    Xts2 = both_count(filename_ts, lines_ts, 3, count_fns)
    names += ['text-count-both-%d' % i for i in xrange(len(count_fns))]

    toc('text counts')
    return ([Xtr1, Xtr2], [Xts1, Xts2], names)


def extract_images_count():
    tic()
    from features.image.imagediff import diff_image_count
    Xtr = diff_image_count(filename_tr, lines_tr)
    Xts = diff_image_count(filename_ts, lines_ts)
    toc('images count')
    return ([Xtr], [Xts], ['image-count-diff', 'image-count-both'])


def extract_brands():
    tic()
    _myreader_tr = myreader_tr.copy()
    _myreader_ts = myreader_ts.copy()

    from features.text.terms import Brands
    m1 = Brands(2)
    Xtr1 = m1.transform(_myreader_tr, lines_tr)
    Xts1 = m1.transform(_myreader_ts, lines_ts)
    m2 = Brands(3)
    Xtr2 = m2.transform(_myreader_tr, lines_tr)
    Xts2 = m2.transform(_myreader_ts, lines_ts)
    toc('brands')
    return ([Xtr1, Xtr2], [Xts1, Xts2],
            ['brands-title-dist', 'brands-descr-dist'])


def extract_topics():
    tic()
    _myreader_tr = myreader_tr.copy()
    _myreader_ts = myreader_ts.copy()

    from features.text.terms import Topics
    m = Topics(3)
    Xtr = m.transform(_myreader_tr, lines_tr)
    Xts = m.transform(_myreader_ts, lines_ts)
    names = ['topic-dist']
    toc('topics')
    return ([Xtr], [Xts], names)


def extract_json():
    tic()
    _myreader_tr = myreader_tr.copy()
    _myreader_ts = myreader_ts.copy()

    from features.text.json import MyJSON
    m = MyJSON()
    Xtr = m.transform(_myreader_tr, lines_tr)
    Xts = m.transform(_myreader_ts, lines_ts)
    names = ['json-dist']
    toc('json')
    return ([Xtr], [Xts], names)


def extract_images_hash():
    if os.path.exists('../data/images/Images_9'):
        from features.image.imagediff import diff_image_hash
        tic()
        Xtr = diff_image_hash(filename_tr, lines_tr)
        Xts = diff_image_hash(filename_ts, lines_ts)
        toc('images hash')
        return ([Xtr], [Xts], ['image-hash-diff'])
    else:
        print 'Warning: images not found'
        return ([], [], [])

import multiprocessing
pool = multiprocessing.Pool(4)

res = [
    pool.apply_async(extract_images_hash),
    pool.apply_async(extract_topics),
    pool.apply_async(extract_brands),
    pool.apply_async(extract_json),
    pool.apply_async(extract_text_expressions),
    pool.apply_async(extract_text_counts),
    pool.apply_async(extract_images_count),
    pool.apply_async(extract_categories),
    pool.apply_async(extract_attributes),
]

Xtr = []
Xts = []
names = []
for r in res:
    _Xtr, _Xts, _names = r.get()
    Xtr += _Xtr
    Xts += _Xts
    names += _names

for i in xrange(len(Xtr)):  # ensure all are matrices
    if len(Xtr[i].shape) == 1:
        Xtr[i] = np.vstack(Xtr[i])
        Xts[i] = np.vstack(Xts[i])
Xtr = np.concatenate(Xtr, 1)
Xts = np.concatenate(Xts, 1)
assert Xtr.shape[1] == len(names)

# create model and validate

print '== model =='

USE_XGBOOST = True

if USE_XGBOOST:
    import xgboost as xgb

    # TODO:
    # - see if applying weights improves AUC since datset is imbalance
    #   (see scale_pos_weight)
    # - reg_alpha and reg_lambda might be interesting parameters
    # - see early_stopping

    # see parameters here:
    # https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
    params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc', 'max_depth': 22,
        'eta': 0.3, 'subsample': 0.6,
        'colsample_bytree': 0.2, 'silent': 0,
    }

    # xgboost does not like spaces in feature_names
    names = [name.replace(' ', '-') for name in names]

    # grid search
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

    best_max_depth = 0
    best_score = 0
    FOLDS = 3

    tic()
    for max_depth in xrange(5, 20+1):
        score = 0
        for tr, ts in StratifiedKFold(ytr, FOLDS):
            xgb_tr = xgb.DMatrix(Xtr[tr], ytr[tr])
            xgb_ts = xgb.DMatrix(Xtr[ts])
            params['max_depth'] = max_depth
            m = xgb.train(params, xgb_tr, 250)
            pp = m.predict(xgb_ts)
            score += roc_auc_score(ytr[ts], pp) / float(FOLDS)
        if score > best_score:
            best_score = score
            best_max_depth = max_depth
    toc('grid search')

    print 'best max_depth: %6d' % best_max_depth
    params['max_depth'] = best_max_depth

    xgb_tr = xgb.DMatrix(Xtr, ytr, feature_names=names)
    xgb_ts = xgb.DMatrix(Xts, feature_names=names)

    m = xgb.train(params, xgb_tr, 260, verbose_eval=True)
    toc('final model')

    pp = m.predict(xgb_ts)
    yp = pp >= 0.5
    toc('predictions')

    xgb.plot_importance(m)
    xgb.plot_tree(m)

else:  # sklearn RandomForest code
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.grid_search import GridSearchCV

    def kaggle_score(m, X, y):
        return roc_auc_score(y, m.predict_proba(X)[:, 1])

    tic()
    m = RandomForestClassifier(250)
    # find a better max_depth if you can...
    m = GridSearchCV(m, {'max_depth': range(15, 28+1)}, kaggle_score,
                     n_jobs=-1)
    m.fit(Xtr, ytr)
    toc('train')
    pp = m.predict_proba(Xts)[:, 1]
    yp = pp >= 0.5
    toc('prediction')

    if os.path.exists('/usr/bin/dot'):  # is graphviz installed?
        from sklearn.tree import DecisionTreeClassifier, export_graphviz
        m = DecisionTreeClassifier(min_samples_leaf=20)
        m.fit(Xtr, ytr)
        export_graphviz(m, feature_names=names,
                        class_names=['non-duplicate', 'duplicate'],
                        label='none', impurity=False, filled=True)
        os.system('dot -Tpdf tree.dot -o tree.pdf')  # compile dot file
        os.remove('tree.dot')

if FINAL_SUBMISSION:
    import datetime
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')
    scores = np.c_[np.arange(pp), pp]
    np.savetxt('../out/vilab-submission-%s.csv' % timestamp, scores,
               delimiter=',', header='id,probability', comments='')
else:
    print 'baseline: %.4f' % (np.sum(yts == 0)/float(len(ts)))
    print 'y=0 | TN=%.2f | FP=%.2f |\ny=1 | FN=%.2f | TP=%.2f |' % (1, 0, 1, 0)
    print

    print 'our model: %.4f' % accuracy_score(yts, yp)
    (TN, FP), (FN, TP) = confusion_matrix(yts, yp)
    print 'y=0 | TN=%.2f | FP=%.2f |\ny=1 | FN=%.2f | TP=%.2f |' % (
        TN / float(np.sum(yts == 0)), FP / float(np.sum(yts == 0)),
        FN / float(np.sum(yts == 1)), TP / float(np.sum(yts == 1)))

    print
    print 'kaggle score:', roc_auc_score(yts, pp)
