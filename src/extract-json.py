# -*- coding: utf-8 -*-


def fn(filename, myreader, info, lines):
    _myreader = myreader.copy()

    from features.text.json import MyJSON
    X = MyJSON().transform(_myreader, lines)
    names = ['json-dist']
    return ([X], names)
