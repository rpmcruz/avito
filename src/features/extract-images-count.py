# -*- coding: utf-8 -*-


def fn(filename, myreader, info, lines):
    from features.image.imagediff import diff_image_count
    X = diff_image_count(filename, lines)
    return ([X], ['image-count-diff', 'image-count-both'])
