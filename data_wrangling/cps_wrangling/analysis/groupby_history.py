# -*- coding: utf-8 -*-
"""
Group the panel dataset according to employment status.

We can group those earning in periods MIS 4 and 8 by their
labor status over the last 3 months.
"""
import json

import arrow
import pandas as pd

with open('../panel_construction/settings.txt') as f:
    settings = json.load(f)

panel_store = pd.HDFStore(settings['panel_store_path'])

m0 = arrow.get('1996-12', 'YYYY-MM')
mn = arrow.get(panel_store.keys()[-1], '/mYY_MM')
months = [x.strftime('%Y_%m') for x in arrow.Arrow.range('month', m0, mn)]

month = months[0]
wp = panel_store.select('m' + month)

wp = wp.loc[:, ((wp['age'] >= 22) & (wp['age'] <= 65)).any(1)]


def is_recently_unemployed(wp, month='both'):
    """
    Checks each row for unemployed/nonemployed in the last 3 months.

    month can be 4, 8, or 'both'.

    (I wonder if groupby().filter could handle this...)
    Returns
    -------

    [ { labor_stats : DataFrame } ]
    A list (MIS=4, 8) of dicts of labor status to DataFrames containing just
    that group.
    """
    if month == 'both':
        months = [4, 8]
    elif month in (4, '4'):
        months == [4]
    elif month in (8, '8'):
        months = [8]
    else:
        raise ValueError
    df = wp['labor_status']
    es = [df[df[x].isin([1, 2])] for x in months]
    employed_idx = [x[x.loc[:, 1:3].isin([1, 2]).all(1)].index for x in es]
    unemployed_idx = [x[x.loc[:, 1:3].isin([3, 4]).all(1)].index for x in es]
    nonemployed_idx = [x[x.loc[:, 1:3].isin([5, 6, 7]).all(1)].index for x in es]

    idxes = zip(['employed', 'unemployed', 'nonemployed'],
                [employed_idx, unemployed_idx, nonemployed_idx])
    dfs = [{k: wp.loc[:, idx[i], m] for k, idx in idxes} for i, m in
           enumerate(months)]
    return dfs


class Runner(object):
    """
    Apply a function to each month in a store and aggregate the results
    """

    def __init__(self, store, apply_func, agg_func, *args, **kwargs):
        """
        Function should take an NDFrame as an argument and return
        something...
        """
        if isinstance(store, pd.io.pytables.HDFStore):
            self.store = store
        else:
            self.store = pd.HDFStore(store)
        self.keys = self.store.keys()
        m0 = arrow.get(self.keys[0], '/mYY-MM')
        mn = arrow.get(self.keys[-1], '/mYY_MM')
        self.months = [x.strftime('%Y_%m') for x in arrow.Arrow.range('month',
                                                                      m0, mn)]
        self.apply_func = apply_func
        self.agg_func = agg_func
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        self.results = [self.apply_func(self.store.select('m' + month)) for
                        month in self.months]
        self.aggerated = [self.agg_func(x) for x in self.results]
        return self.aggregated
