# -*- coding: utf-8 -*-


def fn(filename, myreader, info, lines):
    _myreader = myreader.copy()

    from features.text.terms import Topics
    X = Topics(3).transform(_myreader, lines)
    names = ['topic-dist']
    return ([X], names)
