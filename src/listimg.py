# -*- coding: utf-8 -*-

import sys
sys.dont_write_bytecode = True
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ioff()
from PIL import Image
import imagehash

pairs = np.genfromtxt('../data/ItemPairs_train.csv', int, delimiter=',',
                      skip_header=1, usecols=(0, 1, 2))
# shuffle
pairs = pairs[np.random.choice(np.arange(len(pairs)), 20, False)]

info = pd.read_csv('../data/ItemInfo_train.csv', dtype={'itemID': int},
                   usecols=(0, 2, 4), index_col=0)

hashes_fn = [
    ('dhash 6', lambda img: imagehash.dhash(img, 6)),
    ('dhash 8', lambda img: imagehash.dhash(img, 8)),
    ('dhash 12', lambda img: imagehash.dhash(img, 12)),
    ('phash 6', lambda img: imagehash.phash(img, 6)),
    ('phash 8', lambda img: imagehash.phash(img, 8)),
    ('phash 12', lambda img: imagehash.phash(img, 12)),
]

for i1, i2, dup in pairs:
    print '-------------------------'
    print
    print dup
    title1 = info.ix[i1]['title']
    title2 = info.ix[i2]['title']
    print title1, 'vs', title2

    a1 = info.ix[i1]['images_array']
    a2 = info.ix[i2]['images_array']
    if type(a1) == str and type(a2) == str:
        t1 = a1.split(', ')
        t2 = a2.split(', ')
        if t1 != [''] and t2 != ['']:
            rows = max(len(t1), len(t2))
            hashes1 = []
            hashes2 = []
            for k in xrange(2):
                arr = (t1, t2)[k]
                for i, image in enumerate(arr):
                    plt.subplot(rows, 2, 1+i*2+k)

                    dirname = image[-1]
                    if image[-2] != '0':
                        dirname = image[-2] + dirname
                    filename = '../data/images/Images_%s/%s/%s.jpg' % (
                        image[-2], dirname, image.lstrip('0'))

                    img = Image.open(filename)
                    hs = [fn(img) for _, fn in hashes_fn]
                    (hashes1, hashes2)[k].append(hs)
                    plt.imshow(img)
            plt.show()
            mins = np.repeat(np.inf, len(hashes_fn))
            for h in xrange(len(hashes_fn)):
                for i in xrange(len(hashes1)):
                    for j in xrange(len(hashes2)):
                        mins[h] = min(mins[h], hashes1[i][h]-hashes2[j][h])
            for (name, _), min_ in itertools.izip(hashes_fn, mins):
                print '%-10s %.2f' % (name, min_)
