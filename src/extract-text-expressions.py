# -*- coding: utf-8 -*-

from utils.tictoc import tic, toc


def fn(filename, myreader, info, lines):
    tic()
    _myreader = myreader.copy()

    from features.text.expressions import StartsWith
    X = StartsWith(3).transform(_myreader, lines)

    names = ['common-start']
    toc('text expressions')
    return ([X], names)
