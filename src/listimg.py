# -*- coding: utf-8 -*-

import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ioff()
import matplotlib.image as mpimg

pairs = np.genfromtxt('../data/ItemPairs_train.csv', int, delimiter=',',
                      skip_header=1, usecols=(0, 1, 2))
# shuffle
pairs = pairs[np.random.choice(np.arange(len(pairs)), len(pairs), False)]

info = pd.read_csv('../data/ItemInfo_train.csv', dtype={'itemID': int},
                   usecols=(0, 2, 4), index_col=0)

for i1, i2, dup in pairs:
    if dup:
        t1 = info.ix[i1]['images_array'].split(', ')
        t2 = info.ix[i2]['images_array'].split(', ')
        if t1 != [''] and t2 != ['']:
            title1 = info.ix[i1]['title']
            title2 = info.ix[i2]['title']
            print title1, 'vs', title2

            rows = max(len(t1), len(t2))
            for k in xrange(2):
                arr = (t1, t2)[k]
                for i, image in enumerate(arr):
                    plt.subplot(2, rows, 1+i+k*rows)

                    dirname = image[-1]
                    if image[-2] != '0':
                        dirname = image[-2] + dirname
                    filename = '../data/images/Images_%s/%s/%s.jpg' % (
                        image[-2], dirname, image.lstrip('0'))

                    img = mpimg.imread(filename)
                    plt.imshow(img)
                plt.title((title1, title2)[k])
            plt.show()
