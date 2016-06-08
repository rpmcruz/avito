# -*- coding: utf-8 -*-

import sys
sys.dont_write_bytecode = True
import pandas as pd
import numpy as np

FIELD = 'images_array'

info = pd.read_csv('../data/ItemInfo_train.csv', dtype={'itemID': int},
                   usecols=(0, 1, 2, 4, 5), index_col=0)

# NOTA: estou a ler apenas as primeiras N linhas
pairs = np.genfromtxt('../data/ItemPairs_train.csv', int, delimiter=',',
                      skip_header=1, usecols=(0, 1, 2))
pairs = pairs[np.random.choice(np.arange(len(pairs)), 50, False)]

for i1, i2, dup in pairs:
    t1 = info.ix[i1][FIELD]
    t2 = info.ix[i2][FIELD]
    #print dup, t1, 'vs', t2
    print '%d\n%s\nVS\n%s\n\n' % (dup, t1, t2)
