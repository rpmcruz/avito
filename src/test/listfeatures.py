# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
plt.ioff()

DENSITY = True

filename = '../../out/y.csv'
y = np.genfromtxt(filename, int, delimiter=',')

for file in sorted(os.listdir('../../out')):
    if file.startswith('features-') and file.endswith('-train.csv'):
        filename = os.path.join('../../out', file)
        try:
            X = np.genfromtxt(filename, float, delimiter=',', names=True)
        except ValueError as ex:
            print 'Error: could not open: %s' % file
            continue
        for name in X.dtype.names:
            x = X[name]
            if np.sum(np.logical_or(x == 0, x == 1)) == len(x):
                p0 = np.sum(x[y == 0] == 1) / float(np.sum(y == 0))
                p1 = np.sum(x[y == 1] == 1) / float(np.sum(y == 1))
                plt.bar((0, 1), (p0, p1), color=('blue', 'red'),
                        tick_label=('y=0', 'y=1'), align='center')
                plt.ylim(0, 1)
                plt.title(name + ' = 1')
            else:
                for dup in (0, 1):
                    color = ('blue', 'red')[dup]
                    label = 'dup = ' + str(dup)
                    if DENSITY:
                        kde = stats.kde.gaussian_kde(x[y == dup])
                        _x = np.linspace(np.amin(x), np.amax(x), 100)
                        plt.plot(_x, kde(_x), color=color, label=label)
                        plt.legend()
                    else:  # histogram
                        plt.hist(x[y == dup], 10, normed=True, color=color,
                                 label=label, alpha=0.3)
                plt.title(name)
            plt.show()
