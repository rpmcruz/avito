#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sys
sys.dont_write_bytecode = True
import os
import numpy as np

N = 50000

y = np.genfromtxt('../out/y.csv', int, max_rows=N)

X = []
files = os.listdir('../out')
for i, f in enumerate(files):
    if f.startswith('features-') and f.endswith('-train.csv'):
        sys.stdout.write('\r%2d%%' % ((100*i)/len(files)))
        filename = os.path.join('../out', f)
        _X = np.genfromtxt(filename, delimiter=',', skip_header=1, max_rows=N)
        if len(_X.shape) == 1:
            _X = _X[:, np.newaxis]
        X.append(_X)
sys.stdout.write('\r            \r')
X = np.concatenate(X, 1)

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb  # to install xgboost: sudo -H pip install --pre xgboost


def kaggle_score(m, X, y):
    return roc_auc_score(y, m.predict_proba(X)[:, 1])


params = {'max_depth': range(15, 25+1)}

print 'RandomForest'
m = GridSearchCV(
    RandomForestClassifier(100), params, kaggle_score, refit=False, n_jobs=-1)
for line in sorted(m.fit(X, y).grid_scores_, key=lambda x: -x[1]):
    print line
print

print 'XGradientBoosting'
m = GridSearchCV(
    xgb.XGBClassifier(n_estimators=100), params, kaggle_score)
for line in sorted(m.fit(X, y).grid_scores_, key=lambda x: -x[1]):
    print line
print

import matplotlib.pyplot as plt
plt.ioff()
xgb.plot_importance(m.best_estimator_)
plt.savefig('xgb-features.pdf')
plt.show()
xgb.plot_tree(m.best_estimator_)
plt.savefig('xgb-tree.pdf', dpi=900)
plt.show()
