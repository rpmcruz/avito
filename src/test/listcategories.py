# -*- coding: utf-8 -*-

import sys
sys.dont_write_bytecode = True

from googleapiclient.discovery import build
service = build(
    'translate', 'v2', developerKey='AIzaSyB2y8w1jnsrrjXB1mG5DcdcHw_Zch8mmqs')


def translate(text):
    return service.translations().list(
        source='ru', target='en', q=text).execute() \
        ['translations'][0]['translatedText']


import pandas as pd
Xinfo = pd.read_csv('../../data/ItemInfo_train.csv')

for i in Xinfo['categoryID'].unique():
    items = Xinfo[Xinfo['categoryID'] == i]
    if len(items):
        print i
        tr = list(items[0:min(len(items), 20)]['title'])
        for t in tr:
            try:
                print translate(t), t
            except Exception:
                pass
        print
