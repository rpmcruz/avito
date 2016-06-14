#!/usr/bin/env python2
# -*- coding: utf-8 -*-

'''
Synchronize whatever features have not been extracted. Creates features under
out/ whenever they are outdated relative to extract-*.py
'''

import sys
sys.dont_write_bytecode = True
import os
import itertools
import numpy as np
import pandas as pd
import importlib
#import multiprocessing
from utils.mycorpus import MyCSVReader
from utils.tictoc import tic, toc


def sync_extract(module, csv, params):
    create = not os.path.exists(csv)
    if not create:
        m1 = os.path.getmtime(csv)
        m2 = os.path.getmtime('features/' + module + '.py')
        create = m2 > m1
    if create:
        tic()
        i = importlib.import_module('features.' + module)
        X, names = i.fn(*params)
        toc(module[8:])
        if len(X):
            if len(X[0].shape) == 1:
                X = [x[:, np.newaxis] for x in X]
            X = np.concatenate(X, 1)
            assert X.shape[1] == len(names)
            #names = ['"' + name + '"' for name in names]
            header = ','.join(names)
            fmt = '%d' if X.dtype == int else '%.6f'
            np.savetxt(csv, X, fmt, delimiter=',', header=header, comments='')


def extract(info_filename, pairs_filename, mode):
    info_filename = os.path.join('../data', info_filename)
    pairs_filename = os.path.join('../data', pairs_filename)

    tic()
    info_df = pd.read_csv(
        info_filename,
        dtype={'itemID': int, 'categoryID': int, 'price': float},
        usecols=(0, 1, 6, 7, 8, 9, 10), index_col=0)
    info_df['line'] = np.arange(len(info_df), dtype=int)
    toc('info file')

    info_reader = MyCSVReader(info_filename)
    toc('info reader')

    cols = (0, 1) if mode == 'train' else (1, 2)
    pairs = np.genfromtxt(pairs_filename, int, delimiter=',', skip_header=1,
                          usecols=cols)
    toc('pairs file')

    # transforma ItemID em linhas do ficheiro CSV e da matriz info
    a = info_df.ix[pairs[:, 0]]['line']
    b = info_df.ix[pairs[:, 1]]['line']
    pairs_lines = np.c_[a, b]
    toc('pairs lines')

    params = (info_filename, info_reader, info_df, pairs_lines)
    modules = [module[:-3] for module in sorted(os.listdir('features'))
               if module.startswith('extract-')]
    csvs = ['../out/features-%s-%s.csv' % (module[8:], mode)
            for module in modules]

    # create features from modules that have been created or changed
    #pool = multiprocessing.Pool(multiprocessing.cpu_count()/2)
    #res = []
    for module, csv in itertools.izip(modules, csvs):
        #res.append(pool.apply_async(sync_extract, (module, csv, params)))
        sync_extract(module, csv, params)
    #for r in res:
    #    r.get()

    # remove whatever has been created by extiguish modules
    vestiges = [os.path.join('../out', f) for f in os.listdir('../out')
                if f.startswith('features-') and f.endswith('-%s.csv' % mode)]
    for v in vestiges:
        if v not in csvs:
            print 'removing old %s...' % v
            os.remove(v)

extract('ItemInfo_train.csv', 'ItemPairs_train.csv', 'train')
print '---------------------------'
extract('ItemInfo_test.csv', 'ItemPairs_test.csv', 'test')

if not os.path.exists('../out/y.csv'):
    pairs_filename = '../data/ItemPairs_train.csv'
    y = np.genfromtxt(pairs_filename, int, delimiter=',', skip_header=1,
                      usecols=[2])
    np.savetxt('../out/y.csv', y, '%d')
