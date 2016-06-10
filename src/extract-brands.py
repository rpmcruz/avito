# -*- coding: utf-8 -*-


def fn(filename, myreader, info, lines):
    _myreader = myreader.copy()

    from features.text.terms import Brands
    X1 = Brands(2).transform(_myreader, lines)
    X2 = Brands(3).transform(_myreader, lines)
    return ([X1, X2], ['brands-title-dist', 'brands-descr-dist'])
