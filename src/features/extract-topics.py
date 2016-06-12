# -*- coding: utf-8 -*-


def fn(filename, myreader, info, lines):
    _myreader = myreader.copy()

    from features.text.terms import Topics
    X1 = Topics(2).transform(_myreader, lines)
    X2 = Topics(3).transform(_myreader, lines)
    return ([X1, X2], ['topics-title-dist', 'topics-descr-dist'])
