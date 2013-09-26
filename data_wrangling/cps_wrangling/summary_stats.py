from __future__ import division

import json

import numpy as np
import pandas as pd


def difference_quantiles(s1, s2, nquantiles=10):
    xs = 1 / nquantiles
    quantiles = np.arange(xs, 1 + xs, xs)

    c1 = pd.cut(s1, [s1.quantile(x) for x in quantiles])  # bin by start earn
    return (s1 - s2).groupby(c1)


def load_earn_store():
    with open('settings.txt', 'r') as f:
        d = json.load(f)

    earn_store_path = d['earn_store_path']
    earn_store = pd.HDFStore(earn_store_path)

    panels = (earn_store.select(k) for k, _ in earn_store.iteritems() if
              len(earn_store.select(k).A) > 0)

    return panels


def get_mean_difference(panels):

    mean_diffs = {}
    for pan in panels:

        try:
            s1 = pan.A['PRERNWA']
        except KeyError:
            s1 = pan.A['PTERNWA']
        try:
            s2 = pan.B['PRERNWA']
        except KeyError:
            s2 = pan.B['PTERNWA']

        s1 = s1.convert_objects(convert_numeric=True)
        s2 = s2.convert_objects(convert_numeric=True)

        mean_diff = difference_quantiles(s1, s2).mean()
        mean_diffs[k] = mean_diff

        return mean_diffs


if __name__ == '__main__':
    means = get_mean_difference(load_earn_store())
