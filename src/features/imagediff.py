# -*- coding: utf-8 -*-

# Extract non-Russian words (except Russian colors)
# This was thought out mainly for the phones category.

import preprocess
import numpy as np
import itertools
from PIL import Image
import imagehash

HASH_SIZE = 8


def diff_image_hash(rows):
    rows, ix = np.unique(rows.flatten('F'), return_inverse=True)
    corpus = preprocess.Documents('../data/ItemInfo_train.csv', 4, rows)
    hashes = [[] for _ in xrange(len(rows))]
    for i, text in enumerate(corpus):
        images = text.split(', ')
        if images != ['']:
            for image in images:
                filename = '../data/Images_%s/%s/%s.jpg' % (
                    image[-2], image[-1], image.lstrip('0'))
                h = imagehash.dhash(Image.open(filename), HASH_SIZE)
                hashes[i].append(h)
    diff = np.ones(len(ix)/2, int)*-1000
    for i, (ix1, ix2) in enumerate(
            itertools.izip(ix[:(len(ix)/2)], ix[(len(ix)/2):])):
        hs1 = hashes[ix1]
        hs2 = hashes[ix2]
        mindist = np.inf
        for h1 in hs1:
            for h2 in hs2:
                mindist = min(mindist, abs(h1 - h2))
        if mindist != np.inf:
            diff[i] = mindist
    return diff
