# -*- coding: utf-8 -*-

from utils.tictoc import tic, toc


def fn(filename, myreader, info, lines):
    tic()
    from features.text.count import diff_count, both_count
    # symbols tested that were not useful: +, *, 1), a)
    count_fns = [
        lambda text: text.count('.'),  # 1
        lambda text: text.count('!'),  # 2
        lambda text: text.count('_'),  # 6
        lambda text: text.count('='),  # 9
        lambda text: text.count(u'•'),  # 10
        lambda text: len(text),  # 11
    ]
    X1 = diff_count(filename, lines, 3, count_fns)
    names = ['text-count-diff-%d' % i for i in xrange(len(count_fns))]

    count_fns = [
        lambda text: text.count(','),  # 0
        lambda text: text.count('-'),  # 3
    ]
    X2 = both_count(filename, lines, 3, count_fns)
    names += ['text-count-both-%d' % i for i in xrange(len(count_fns))]

    toc('text counts')
    return ([X1, X2], names)
