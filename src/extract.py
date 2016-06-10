#!/usr/bin/env python2
# -*- coding: utf-8 -*-

'''
Synchronize whatever features have not been extracted. Creates features under
out/ whenever they are outdated relative to extract-*.py
'''

import sys
sys.dont_write_bytecode = True
import os
import numpy as np
import pandas as pd
import importlib
from utils.mycorpus import MyCSVReader
from utils.tictoc import tic, toc


def extract(info_filename, pairs_filename, mode):
    info_filename = os.path.join('../data', info_filename)
    pairs_filename = os.path.join('../data', pairs_filename)

    tic()
    info = pd.read_csv(
        info_filename,
        dtype={'itemID': int, 'categoryID': int, 'price': float},
        usecols=(0, 1, 6, 7, 8, 9, 10), index_col=0)
    info['line'] = np.arange(len(info), dtype=int)
    toc('item info')

    myreader = MyCSVReader(info_filename)
    toc('lines to seek')

    cols = (0, 1) if mode == 'train' else (1, 2)
    pairs = np.genfromtxt(pairs_filename, int, delimiter=',', skip_header=1,
                          usecols=cols)

    # transforma ItemID em linhas do ficheiro CSV e da matriz info
    lines = np.asarray(
        [(info.ix[i1]['line'], info.ix[i2]['line'])
         for i1, i2, d in pairs], int)
    toc('pairs to lines')

    for filename in os.listdir('.'):
        if filename.startswith('extract-'):
            csv = '../out/%s-%s.csv' % (filename[:-3], mode)
            create = not os.path.exists(csv)
            if not create:
                m1 = os.path.getmtime(csv)
                m2 = os.path.getmtime(filename)
                create = m2 > m1
            if create:
                i = importlib.import_module(filename[:-3])
                X, names = i.fn(info_filename, myreader, info, lines)
                header = ','.join(names)
                np.savetxt(csv, X, delimiter=',', header=header, comments='')

extract('ItemInfo_train.csv', 'ItemPairs_train.csv', 'train')
extract('ItemInfo_test.csv', 'ItemPairs_test.csv', 'test')
