#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sys
sys.dont_write_bytecode = True
import os
import csv
import numpy as np
from utils.tictoc import tic, toc

print 'read data...'
y = np.genfromtxt('../out/y.csv', int)


def read_data(f):
    print f
    filename = os.path.join('../out', f)
    X = np.genfromtxt(filename, delimiter=',', skip_header=1)
    if len(X.shape) == 1:
        X = X[:, np.newaxis]
    with open(filename, 'rb') as f:
        header = next(csv.reader(f, quotechar='"'))
    return (header, X)

tic()
files = sorted(os.listdir('../out'))
files = [f for f in files
         if f.startswith('features-') and f.endswith('-train.csv')]
names, X = zip(*map(read_data, files))
toc('read data')
X = np.concatenate(X, 1)

from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb  # to install xgboost: sudo -H pip install --pre xgboost


def kaggle_score(m, X, y):
    return roc_auc_score(y, m.predict_proba(X)[:, 1])

params = {'max_depth': range(15, 25+1)}
Xtr, Xts, ytr, yts = train_test_split(X, y)

print 'testing...'

subsample = 0.05
subcolsample = np.sqrt(X.shape[1]) / X.shape[1]

for max_depth in xrange(26, 40+1):
    #m = RandomForestClassifier(100, max_depth=max_depth)
    tic()
    m = xgb.XGBClassifier(n_estimators=100, max_depth=max_depth, subsample=0.1,
                          colsample_bytree=subcolsample)
    m.fit(Xtr, ytr)
    pp = m.predict_proba(Xts)[:, 1]
    toc('%2d %.4f' % (max_depth, roc_auc_score(yts, pp)))
    sys.stdout.flush()

import matplotlib.pyplot as plt
plt.ioff()
xgb.plot_importance(m.best_estimator_)
plt.savefig('xgb-features.pdf')
plt.show()

xgb.plot_tree(m)
plt.savefig('xgb-tree.pdf', dpi=900)
plt.show()
