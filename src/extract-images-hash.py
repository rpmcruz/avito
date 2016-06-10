# -*- coding: utf-8 -*-

import os
from utils.tictoc import tic, toc


def fn(filename, myreader, info, lines):
    if os.path.exists('../data/images/Images_9'):
        from features.image.imagediff import diff_image_hash
        tic()
        X = diff_image_hash(filename, lines)
        toc('images hash')
        return ([X], ['image-hash-diff'])
    else:
        print 'Warning: images not found'
        return ([], [])
