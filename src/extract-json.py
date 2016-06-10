# -*- coding: utf-8 -*-

from utils.tictoc import tic, toc


def fn(filename, myreader, info, lines):
    tic()
    _myreader = myreader.copy()

    from features.text.json import MyJSON
    X = MyJSON().transform(_myreader, lines)
    names = ['json-dist']
    toc('json')
    return ([X], names)
