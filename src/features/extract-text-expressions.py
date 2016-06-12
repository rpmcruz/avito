# -*- coding: utf-8 -*-


def fn(filename, myreader, info, lines):
    _myreader = myreader.copy()

    from features.text.expressions import StartsWith, UnicodeCategories, \
        UNICODE_CATEGORIES
    X1 = StartsWith().transform(_myreader, lines)
    X2 = UnicodeCategories().transform(_myreader, lines)

    names = ['both-start'] + ['both-' + cat for cat in UNICODE_CATEGORIES]
    return ([X1, X2], names)
