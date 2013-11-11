# -*- coding: utf-8 -*-
"""
Group the panel dataset according to employment status.

We can group those earning in periods MIS 4 and 8 by their
labor status over the last 3 months.
"""
import json

import arrow
import numpy as np
import pandas as pd

from helpers import get_useful

with open('../panel_construction/settings.txt') as f:
    settings = json.load(f)

panel_store = pd.HDFStore(settings['panel_store_path'])


def is_recently_unemployed(wp, month='both'):
    """
    Checks each row for unemployed/nonemployed in the last 3 months.

    month can be 4, 8, or 'both'.

    (I wonder if groupby().filter could handle this...)
    Returns
    -------

    {timestamp: { labor_stats : DataFrame } }

    A list (MIS=4, 8) of dicts of labor status to DataFrames containing just
    that group.

    aggfuncs should be able to ignore Nones
    """
    wp = get_useful(wp)
    wp = wp.loc[:, ((wp['age'] >= 22) & (wp['age'] <= 65)).any(1)]

    if month == 'both':
        months = [4, 8]
    elif month in (4, '4'):
        months == [4]
    elif month in (8, '8'):
        months = [8]
    else:
        raise ValueError

    if 8 not in wp.minor_axis:
        months = [4]
    if 4 not in wp.minor_axis:
        return None

    df = wp['labor_status']
    es = [df[df[x].isin([1, 2])] for x in months]
    employed_idx = [x[x.loc[:, 1:3].isin([1, 2]).all(1)].index for x in es]
    unemployed_idx = [x[x.loc[:, 1:3].isin([3, 4]).all(1)].index for x in es]
    nonemployed_idx = [x[x.loc[:, 1:3].isin([5, 6, 7]).all(1)].index for x in es]

    idxes = zip(['employed', 'unemployed', 'nonemployed'],
                [employed_idx, unemployed_idx, nonemployed_idx])
    res = {}
    for i, m in enumerate(months):
        stamp = pd.Timestamp(pd.datetime(int(wp['year'][m].dropna().values[0]),
                                         int(wp['month'][m].dropna().values[0]), 1))
        res[stamp] = {k: wp.loc[:, idx[i], m] for k, idx in idxes}
    return res


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
        self.months = self._generate_months()
        self.apply_func = apply_func
        self.agg_func = agg_func
        self._args = args
        self._kwargs = kwargs

    def _generate_months(self):
        m0 = arrow.get(self.keys[0], '/mYY-MM')
        mn = arrow.get(self.keys[-1], '/mYY_MM')
        months = (x.strftime('%Y_%m') for x in arrow.Arrow.range('month', m0, mn))
        months = (month for month in months if '/m' + month in self.keys)
        for month in months:
            yield month

    def __call__(self):
        self.results = {month: self.apply_func(self.store.select('m' + month))
                        for month in self.months}

        self.aggregated = self.intermediate_aggregator(self)
        return self.aggregated

    def write(self, path):
        self.aggregated.to_csv(path)

    def intermediate_aggregator(self):
        """
        Any given month can have values coming from two Panels (MIS 4 and 8).
        Aggregate those two here.
        """
        e_agg = []
        u_agg = []
        n_agg = []

        months = filter(lambda x: self.results[x] is not None, self.results)
        for m in months:
            for t in self.results[m]:
                by_type = self.results[m][t]
                e_agg.append({t: by_type['employed']})
                u_agg.append({t: by_type['unemployed']})
                n_agg.append({t: by_type['nonemployed']})

        e_agg = sorted(e_agg, key=lambda k: k.keys())
        u_agg = sorted(u_agg, key=lambda k: k.keys())
        n_agg = sorted(n_agg, key=lambda k: k.keys())

        df_dict = {}
        for name, u_type in zip(['e', 'u', 'n'], [e_agg, u_agg, n_agg]):
            res_dict = {}
            for d in u_type:
                k = d.keys()[0]
                if k in res_dict:
                    res_dict[k] = pd.concat([d[k], res_dict[k]]).sort_index()
                else:
                    res_dict[k] = d[k]
            # wp = pd.Panel(res_dict)
            medians = {}
            for k, df in res_dict.iteritems():
                medians[k] = df.earnings.median()
            df_dict[name] = pd.Series(medians)
        return pd.DataFrame(df_dict)
