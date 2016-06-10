# -*- coding: utf-8 -*-

from utils.tictoc import tic, toc


def fn(filename, myreader, info, lines):
    tic()
    _myreader = myreader.copy()

    from features.text.terms import Brands
    X1 = Brands(2).transform(_myreader, lines)
    X2 = Brands(3).transform(_myreader, lines)
    toc('brands')
    return ([X1, X2], ['brands-title-dist', 'brands-descr-dist'])
