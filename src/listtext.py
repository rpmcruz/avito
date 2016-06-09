# -*- coding: utf-8 -*-

import sys
sys.dont_write_bytecode = True
import pandas as pd
import numpy as np

from googleapiclient.discovery import build
service = build(
    'translate', 'v2', developerKey='AIzaSyB2y8w1jnsrrjXB1mG5DcdcHw_Zch8mmqs')


def translate(text):
    return service.translations().list(
        source='ru', target='en', q=text).execute() \
        ['translations'][0]['translatedText']

info = pd.read_csv('../data/ItemInfo_train.csv', dtype={'itemID': int},
                   usecols=(0, 1, 2, 5), index_col=0)

# NOTA: estou a ler apenas as primeiras N linhas
pairs = np.genfromtxt('../data/ItemPairs_train.csv', int, delimiter=',',
                      skip_header=1, usecols=(0, 1, 2))
pairs = pairs[np.random.choice(np.arange(len(pairs)), 50, False)]

for i1, i2, dup in pairs:
    t1 = info.ix[i1]['attrsJSON']
    t2 = info.ix[i2]['attrsJSON']
    #print dup, t1, 'vs', t2
    print '%d\n%s\nVS\n%s\n\n' % (dup, translate(t1).replace('&quot;', '"'),
                                  translate(t2).replace('&quot;', '"'))
