# -*- coding: utf-8 -*-

from utils.tictoc import tic, toc


def fn(filename, myreader, info, lines):
    tic()
    _myreader = myreader.copy()

    from features.text.terms import Topics
    X = Topics(3).transform(_myreader, lines)
    names = ['topic-dist']
    toc('topics')
    return ([X], names)
