# -*- coding: utf-8 -*-

import string
import enchant
import pandas as pd

Xinfo = pd.read_csv('data/ItemInfo_train.csv')
with open('colors.txt') as f:
    colors = f.readlines()

d = enchant.Dict('ru')
exclude = set(string.punctuation)
for i, title in enumerate(Xinfo['title']):
    t = []
    for word in title.split(' '):
        word = ''.join(ch for ch in word if ch not in exclude)
        if word and (not d.check(word) or word in colors):
            t.append(word)
    if t:
        print ' '.join(t)
    if i == 50:
        break
