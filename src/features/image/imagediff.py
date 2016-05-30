# -*- coding: utf-8 -*-

from utils.mycorpus import MyCorpus
import numpy as np
import itertools
from PIL import Image
import imagehash

HASH_SIZE = 8


def diff_image_hash(rows):
    rows, ix = np.unique(rows.flatten('F'), return_inverse=True)
    corpus = MyCorpus('../data/ItemInfo_train.csv', 4, rows)
    hashes = [[] for _ in xrange(len(rows))]
    for i, text in enumerate(corpus):
        images = text.split(', ')
        if images != ['']:
            for image in images:
                dirname = image[-1]
                if image[-2] != '0':
                    dirname = image[-2] + dirname
                filename = '../data/images/Images_%s/%s/%s.jpg' % (
                    image[-2], dirname, image.lstrip('0'))
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


def diff_image_count(rows):
    rows, ix = np.unique(rows.flatten('F'), return_inverse=True)
    corpus = MyCorpus('../data/ItemInfo_train.csv', 4, rows)
    counts = np.zeros(len(rows))
    for i, text in enumerate(corpus):
        images = text.split(', ')
        if images != ['']:
            counts[i] = len(images)
    c1 = counts[ix[:(len(ix)/2)]]
    c2 = counts[ix[(len(ix)/2):]]
    return np.c_[np.abs(c1 - c2), np.logical_and(c1 > 0, c2 > 0)]
