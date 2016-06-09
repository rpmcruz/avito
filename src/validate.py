# -*- coding: utf-8 -*-

import sys
sys.dont_write_bytecode = True
import os
import pandas as pd
import numpy as np
from utils.mycorpus import MyCSVReader
from utils.tictoc import tic, toc

N = 5000
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
toc()
# NOTA: estou a ler apenas as primeiras N linhas
pairs_tr = np.genfromtxt('../data/ItemPairs_train.csv', int, delimiter=',',
                         skip_header=1, usecols=(0, 1, 2))
if FINAL_SUBMISSION:
    pairs_ts = np.genfromtxt('../data/ItemPairs_test.csv', int, delimiter=',',
                             skip_header=1, usecols=(1, 2))
    ytr = pairs_tr[:, -1]
else:
    pairs_tr = pairs_tr[  # undersample to speedup things
        np.random.choice(np.arange(len(pairs_tr)), 1000, False)]
    # split train into train and test
    idx = np.arange(len(pairs_tr))
    np.random.shuffle(idx)
    tr = idx[:int(0.60*len(pairs_tr))]
    ts = idx[int(0.60*len(pairs_tr)):]
    pairs_ts = pairs_tr[ts]
    pairs_tr = pairs_tr[tr]
    ytr = pairs_tr[:, -1]
    yts = pairs_ts[:, -1]
    pairs_ts = pairs_ts[:, :-1]  # drop dups
toc()

# transforma ItemID em linhas do ficheiro CSV e da matriz info
tic()
lines_tr = np.asarray(
    [(info_tr.ix[i1]['line'], info_tr.ix[i2]['line'])
     for i1, i2, d in pairs_tr], int)
lines_ts = np.asarray(
    [(info_ts.ix[i1]['line'], info_ts.ix[i2]['line'])
     for i1, i2 in pairs_ts], int)
toc()

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
    categories01_tr = encoding.fit_transform(categories_tr)
    categories01_ts = encoding.transform(categories_ts)

    df = pd.read_csv('../data/Category.csv', dtype=int, index_col=0)
    encoding = OneHotEncoder(dtype=int, sparse=False)
    parents_tr = df.ix[categories_tr[:, -1]].as_matrix(['parentCategoryID'])
    parents_ts = df.ix[categories_ts[:, -1]].as_matrix(['parentCategoryID'])
    parents01_tr = encoding.fit_transform(parents_tr)
    parents01_ts = encoding.transform(parents_ts)

    from utils.categorias import categorias
    names = [r'\"' + categorias[i].encode('utf8') + r'\"'
             for i in np.unique(categories_tr)]
    names += [r'\"' + categorias[i].encode('utf8') + r'\"'
              for i in np.unique(parents_tr)]
    toc('categories')
    return ([categories01_tr, parents01_tr], [categories01_ts, parents01_ts],
            names)


def extract_attributes():
    tic()
    Xtr = []
    Xts = []
    attrbs = ('price', 'locationID', 'metroID', 'lat', 'lon')
    for attr in attrbs:
        a = info_tr.as_matrix([attr])[:, -1]
        x = np.abs(a[lines_tr[:, 0]] - a[lines_tr[:, 1]])
        x[np.isnan(x)] = 10000  # NaN handling
        Xtr.append(x)

        a = info_ts.as_matrix([attr])[:, -1]
        x = np.abs(a[lines_ts[:, 0]] - a[lines_ts[:, 1]])
        x[np.isnan(x)] = 10000  # NaN handling
        Xts.append(x)
    toc('attributes')
    return (Xtr, Xts, attrbs)


def extract_text_counts():
    from features.text.count import diff_count
    count_fns = [
        lambda text: text.count(','),
        lambda text: text.count('.'),
        lambda text: text.count('!'),
        lambda text: text.count('-'),
        lambda text: text.count('+'),
        lambda text: text.count('*'),
        lambda text: text.count('_'),
        lambda text: text.count('1)'),
        lambda text: text.count('a)'),
        lambda text: text.count('='),
        lambda text: text.count(u'•'),
        lambda text: len(text),
    ]
    Xtr = diff_count(filename_tr, lines_tr, 3, count_fns)
    Xts = diff_count(filename_ts, lines_ts, 3, count_fns)

    names = ['text-count-diff-%d' % i for i in xrange(len(count_fns))]
    names += ['text-count-both-%d' % i for i in xrange(len(count_fns))]
    toc('text counts')
    return ([Xtr], [Xts], names)


def extract_images_count():
    from features.image.imagediff import diff_image_count
    tic()
    Xtr = diff_image_count(filename_tr, lines_tr)
    Xts = diff_image_count(filename_ts, lines_ts)
    toc('images count')
    return ([Xtr], [Xts], ['image-count-diff', 'image-count-both'])


def extract_brands():
    from features.text.brands import Brands
    tic()
    m1 = Brands(2).fit(filename_tr, lines_tr)
    Xtr1 = m1.transform(filename_tr, lines_tr)
    Xts1 = m1.transform(filename_ts, lines_ts)
    m2 = Brands(3).fit(filename_tr, lines_tr)
    Xtr2 = m2.transform(filename_tr, lines_tr)
    Xts2 = m2.transform(filename_ts, lines_ts)
    toc('brands')
    return ([Xtr1, Xtr2], [Xts1, Xts2], ['brand-title', 'brand-descr'])


def extract_topics():
    from features.text.topics import Topics, NTOPICS
    tic()
    m = Topics(3).fit(filename_tr, lines_tr)
    Xtr = m.transform(filename_tr, lines_tr)
    Xts = m.transform(filename_ts, lines_ts)
    names = ['topic-%d' % i for i in xrange(NTOPICS)] + \
        ['topic-dist-cos', 'topic-dist2']
    toc('topics')
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
pool = multiprocessing.Pool(2)

res = [
    pool.apply_async(extract_images_hash),
    pool.apply_async(extract_topics),
    pool.apply_async(extract_text_counts),
    pool.apply_async(extract_images_count),
    pool.apply_async(extract_categories),
    pool.apply_async(extract_attributes),
    pool.apply_async(extract_brands),
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

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV


def kaggle_score(m, X, y):
    return roc_auc_score(y, m.predict_proba(X)[:, 1])

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
    print 'best max_depth: %6d' % best_max_depth
    params['max_depth'] = best_max_depth
    toc('grid search')

    xgb_tr = xgb.DMatrix(Xtr, ytr, feature_names=names)
    xgb_ts = xgb.DMatrix(Xts, feature_names=names)

    m = xgb.train(params, xgb_tr, 260, verbose_eval=True)
    toc('final model')

    pp = m.predict(xgb_ts)
    yp = pp >= 0.5
    toc('predictions')

    import matplotlib.pyplot as plt
    plt.ioff()
    xgb.plot_importance(m)
    plt.savefig('xgb-features.pdf')
    plt.show()
    xgb.plot_tree(m)
    plt.savefig('xgb-tree.pdf', dpi=900)
    plt.show()

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
    print 'best params:', m.best_params_
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
        os.system('dot -Tpdf tree.dot -o sklearn-tree.pdf')  # compile dot file
        os.remove('tree.dot')

if FINAL_SUBMISSION:
    import datetime
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')
    scores = np.c_[np.arange(pp), pp]
    np.savetxt('../out/vilab-submission-%s.csv' % timestamp, scores,
               delimiter=',', header='id,probability', comments='')
else:
    print 'best params:', m.best_params_
    print

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

if os.path.exists('/usr/bin/dot'):  # has graphviz installed?
    from sklearn.tree import DecisionTreeClassifier, export_graphviz
    m = DecisionTreeClassifier(min_samples_leaf=50)
    m.fit(Xtr, ytr)
    export_graphviz(m, feature_names=names,
                    class_names=['non-duplicate', 'duplicate'], label='none',
                    impurity=False, filled=True)
    os.system('dot -Tpdf tree.dot -o tree.pdf')  # compile dot file
    os.remove('tree.dot')
