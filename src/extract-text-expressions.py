# -*- coding: utf-8 -*-


def fn(filename, myreader, info, lines):
    _myreader = myreader.copy()

    from features.text.expressions import StartsWith
    X = StartsWith(3).transform(_myreader, lines)

    names = ['common-start']
    return ([X], names)
