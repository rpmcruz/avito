# -*- coding: utf-8 -*-

import string
import enchant
import pandas as pd

Xinfo = pd.read_csv('data/ItemInfo_train.csv')

d = enchant.Dict('ru')
exclude = set(string.punctuation)
for title in Xinfo['title']:
    t = []
    for word in title.split(' '):
        word = ''.join(ch for ch in word if ch not in exclude)
        if len(word) and not d.check(word):
            t.append(word)
    if len(t):
        print ' '.join(t)
