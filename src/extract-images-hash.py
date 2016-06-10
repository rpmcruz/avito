# -*- coding: utf-8 -*-

import os


def fn(filename, myreader, info, lines):
    if os.path.exists('../data/images/Images_9'):
        from features.image.imagediff import diff_image_hash
        X = diff_image_hash(filename, lines)
        return ([X], ['image-hash-diff'])
    else:
        print 'Warning: images not found'
        return ([], [])
