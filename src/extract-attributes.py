# -*- coding: utf-8 -*-

import numpy as np


def fn(filename, myreader, info, lines):
    X = []
    # not using 'locationID' because it degrades performance
    attrbs = ['price', 'metroID']
    for attr in attrbs:
        a = info.as_matrix([attr])[:, -1]
        x = np.abs(a[lines[:, 0]] - a[lines[:, 1]])
        x[np.isnan(x)] = 10000  # NaN handling
        X.append(x)
    # lat, lon use euler distance
    # using lat,lon individually degrades performance, but this metric
    # seems to improve it slightly
    l1 = info.as_matrix(['lon'])[:, -1]
    l2 = info.as_matrix(['lat'])[:, -1]
    x = (l1[lines[:, 0]] - l2[lines[:, 1]]) ** 2
    x[np.isnan(x)] = 10000  # NaN handling
    X.append(x)
    return (X, attrbs + ['lon-lat'])
