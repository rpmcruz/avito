# -*- coding: utf-8 -*-

import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy import stats

print 'load items info...'
Xinfo = pd.read_csv('data/ItemInfo_train.csv', index_col=0)
print 'load items pairs...'
Xpair = pd.read_csv('data/ItemPairs_train.csv')

# idxmap is an efficient mapping between item-id and row index
# we could also use Xinfo.ix[indices], but this approach seems
# slightly faster
print 'load items mapping...'
if os.path.exists('idxmap.pickle'):
    with open('idxmap.pickle', 'rb') as f:
        idxmap = pickle.load(f)
else:
    lastid = Xinfo.index[-1]
    idxmap = np.zeros(lastid+1, int)
    for i, (idx, item) in enumerate(Xinfo.iterrows()):
        if idx % np.ceil(Xinfo.shape[0]/100.) == 0:
            sys.stdout.write('\r%2d%%' % (100*idx/Xinfo.shape[0]))
        idxmap[idx] = i
    sys.stdout.write('\r            \r')
    with open('idxmap.pickle', 'wb') as f:
        pickle.dump(idxmap, f, pickle.HIGHEST_PROTOCOL)

# compare stuff between articles

import matplotlib.pyplot as plt
plt.ioff()
colors = ('blue', 'red')

dpairs = (
    Xpair[Xpair['isDuplicate'] == 0].as_matrix(['itemID_1', 'itemID_2']),
    Xpair[Xpair['isDuplicate'] == 1].as_matrix(['itemID_1', 'itemID_2']),
)

print 'distances'

geo = Xinfo.as_matrix(['lat', 'lon'])

for density in (False, True):
    for dup in (0, 1):
        pairs = dpairs[dup]
        lat = geo[idxmap[pairs[:, 0]], 0] - geo[idxmap[pairs[:, 1]], 0]
        lon = geo[idxmap[pairs[:, 0]], 1] - geo[idxmap[pairs[:, 1]], 1]
        dists = np.sqrt(lat**2 + lon**2)

        if density:
            kde = stats.kde.gaussian_kde(dists)
            x = np.linspace(0, 80, 100)
            plt.plot(x, kde(x), color=colors[dup], label=str(dup))
            plt.ylim(0, 0.6)
        else:
            plt.hist(dists, 10, (0, 80), normed=True,
                     color=colors[dup], label=str(dup), alpha=0.3)
    plt.legend(title='is duplicate?')
    plt.show()

print 'map positions'

for dup in (0, 1):
    pairs = dpairs[dup].flatten()
    plt.scatter(geo[idxmap[pairs], 0], geo[idxmap[pairs], 1],
                color=colors[dup], s=3)
    plt.title('duplicate? ' + str(dup))
    plt.show()

print 'same metro, location, category'

for var in ('metroID', 'locationID', 'categoryID'):
    loc = Xinfo.as_matrix([var])

    for dup in (0, 1):
        pairs = dpairs[dup]
        same = loc[idxmap[pairs[:, 0]]] == loc[idxmap[pairs[:, 1]]]
        plt.bar((0, 1), (np.sum(same == 1)/float(len(same)),
                         np.sum(same == 0)/float(len(same))),
                color=colors[dup], alpha=0.3, label=str(dup))
    plt.legend(title='is duplicate?')
    plt.xticks((0.5, 1.5), ('same', 'different'))
    plt.title(var)
    plt.ylim(0, 1)
    plt.show()

print 'dups per category'

from categorias import categorias
cat = Xinfo.as_matrix(['categoryID'])
uniquecat = np.unique(cat)
freqs = []

for i in uniquecat:
    n0 = np.sum(cat[idxmap[dpairs[0].flatten()]] == i)
    n1 = np.sum(cat[idxmap[dpairs[1].flatten()]] == i)
    freqs.append(n1 / float(n0+n1))

s = np.argsort(freqs)
freqs = np.asarray(freqs)[s]
uniquecat = uniquecat[s]

plt.barh(np.arange(len(freqs)), freqs)
plt.yticks(np.arange(len(freqs)), [categorias[i] for i in uniquecat],
           fontsize=10)
plt.xlim(0, 1)
plt.ylim(0, len(freqs))
plt.show()

print 'prices'


def diff_price(p0, p1):
    dp = np.abs(p0 - p1)
    dp = dp / np.amin([p0, p1], 0)  # relative difference
    dp = dp[np.logical_and(p0 > 0, p1 > 0)]  # remove zeros
    dp = dp[np.logical_not(np.isnan(dp))]  # remove nans
    #dp = dp[dp < np.percentile(dp, 90)]  # reject outliers
    return dp

prices = Xinfo.as_matrix(['price'])
dprices = [
    diff_price(prices[idxmap[dpairs[dup][:, 0]]],
               prices[idxmap[dpairs[dup][:, 1]]]) for dup in [0, 1]]
plt.boxplot(dprices, labels=[0, 1])
plt.xlabel('is duplicate?')
plt.ylabel('relative price difference')
plt.show()

print 'quantiles'
print '0, 25, 50, 75, 100:', np.percentile(dprices[1], [0, 25, 50, 75, 100])
print '90-100:', np.percentile(dprices[1], np.arange(90, 101))
