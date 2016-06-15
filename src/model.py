#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sys
sys.dont_write_bytecode = True
import os
import csv
import numpy as np
from utils.tictoc import tic, toc


def read_file(f):
    print f
    filename = os.path.join('../out', f)
    X = np.genfromtxt(filename, float, delimiter=',', skip_header=1)
    if len(X.shape) == 1:
        X = X[:, np.newaxis]
    with open(filename, 'rb') as f:
        header = next(csv.reader(f))
    return header, X


def read_files(dtype):
    tic()
    files = sorted(os.listdir('../out'))
    files = [f for f in files
             if f.startswith('features-') and f.endswith('-%s.csv' % dtype)]
    names, X = zip(*map(read_file, files))
    toc('read %s data' % dtype)
    return names, np.concatenate(X, 1)

from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb  # to install xgboost: sudo -H pip install --pre xgboost

FINAL_SUBMISSION = False

print 'read data...'
ytr = np.genfromtxt('../out/y.csv', int)
names, Xtr = read_files('train')
names = [item for sublist in names for item in sublist]

params = {'max_depth': range(15, 25+1)}
if FINAL_SUBMISSION:
    _, Xts = read_files('test')
else:
    Xtr, Xts, ytr, yts = train_test_split(Xtr, ytr)

print 'model...'

# tips:
# http://www.slideshare.net/odsc/owen-zhangopen-sourcetoolsanddscompetitions1
# http://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

subsample = 1  # 1 seems best
gamma = 0

for colsample in [0.55]:
    for max_depth in [60]:  # try bigger
        for learning_rate in [0.08]:
            for min_child_weight in [1]:  # seems fine
                tic()
                m = xgb.XGBClassifier(n_estimators=100,
                                      max_depth=max_depth,
                                      learning_rate=learning_rate,
                                      gamma=gamma,
                                      min_child_weight=min_child_weight,
                                      colsample_bytree=colsample,
                                      subsample=subsample)
                m.fit(Xtr, ytr)
                pp = m.predict_proba(Xts)[:, 1]
                if FINAL_SUBMISSION:
                    import datetime
                    timestamp = datetime.datetime.now().strftime(
                        '%Y-%m-%d-%H:%M')
                    scores = np.c_[np.arange(len(pp)), pp]
                    np.savetxt('../out/vilab-submission-%s.csv' % timestamp,
                               scores, '%d,%.8f', ',', header='id,probability',
                               comments='')
                    toc()
                else:
                    toc('cs=%.2f md=%2d lr=%.2f mcw=%1d g=%d score=%.4f' % (
                        colsample, max_depth, learning_rate, min_child_weight,
                        gamma, roc_auc_score(yts, pp)))
                sys.stdout.flush()

import matplotlib.pyplot as plt
plt.ioff()
xgb.plot_importance(m, tick_label=names)
plt.savefig('xgb-features.pdf')
plt.show()

'''
xgb.plot_tree(m)
plt.savefig('xgb-tree.pdf', dpi=900)
plt.show()
'''
