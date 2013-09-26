"""
Take the HDFStore of monthly cps files and construct the pseudo-panel.

Match based on identifiers and (demographic, etc.).
Earnings questions are asked in MIS 4 and 8 only, so about a quarter
of the full sample should be included in the panel (ideally).
"""
from __future__ import division

import itertools
import json
from time import strftime, strptime, struct_time

import matplotlib.pyplot as plt
import pandas as pd


def get_next_month(this_month):
    """
    str -> str

    Takes a string in mYYYY_MM format and returns the next month,
    in the same format.
    """
    struct = strptime(this_month, 'm%Y_%m')
    new_year = [struct.tm_year + 1]
    new_struct = struct_time(itertools.chain(*[new_year, struct[1:]]))
    new_month = strftime('m%Y_%m', new_struct)
    return new_month


def match_surveys(df1, df2):
    """
    DataFrame -> DataFrame -> Index
    """
    # naiive matching for now.  TODO: Not valid.
    if 'PRERNWA' in df1.columns:
        w_id1 = 'PRERNWA'
    else:
        w_id1 = 'PTERNWA'
    if 'PRERNWA' in df2.columns:
        w_id2 = 'PRERNWA'
    else:
        w_id2 = 'PTERNWA'

    s1 = df1[w_id1][df1[w_id1] > 0]
    s2 = df2[w_id2][df2[w_id2] > 0]

    idx = s1.index.intersection(s2.index)
    return idx


def check_dtypes(wp):
    """
    Ensure that the dtypes are numeric.
    """
    if (wp.A.dtypes == object).all():
        dfa = wp.A.convert_objects(convert_dates=False, convert_numeric=True)
        dfb = wp.B.convert_objects(convert_dates=False, convert_numeric=True)
        return pd.Panel({'A': dfa, 'B': dfb})
    else:
        return wp


def get_earnings_panel(month, store):
    # TODO: table format pan w/ multiIndex isn't going well.  Used fixed for now.
    # TODO: Empty frames for 2004-01 : 2004-04
    """
    #-----------------------------------------------------------------------------
    idx = pd.MultiIndex.from_tuples([('one', 1), ('one', 2), ('two', 1), ('two', 2)])
    df = pd.DataFrame({'A': np.random.randn(4), 'B': np.random.randn(4)}, index=idx)
    #-----------------------------------------------------------------------------

    str -> Panel

    Return panel of a survey wave where the earnings reported are positive.
    """
    pre = '/monthly/data/'
    next_month = get_next_month(month)
    df1 = store.select(pre + month)
    df2 = store.select(pre + next_month)

    idx = match_surveys(df1, df2)
    pan = pd.Panel({'A': df1.loc[idx], 'B': df2.loc[idx]})
    return pan


def make_earn_panel(earn_store):
    """Just for those in MIS 4 and 16 who anser income questions"""
    keys = earn_store.keys()
    wp = pd.Panel({k.lstrip('/'): earn_store.select(k) for k in keys})
    return wp


def make_full_panel(full_store):
    """
    The full panel.  The store layout will be:

        store.month

    where month corresponds to the wave whose first month in
    the survey equals one (MIS=IM=1).  Each node will be a dataframe?
    where

                                        data
        (interview_date, unique_id)

    alternatively a panel where the items are interview_date / MIS.
    The unique_id section of the index should be constant through
    the full 8 interview months, modulo survey attrition.
    """
    


def main():
    with open('settings.txt', 'r') as f:
        settings = json.load(f)

    cstore_path = settings['cstore_path']
    store = pd.HDFStore(cstore_path)
    earn_store_path = settings['earn_store_path']
    earn_store = pd.HDFStore(earn_store_path)
    pre = '/monthly/data/'  # Path inside HDFStore.  Concat to month

    keys = store.keys()
    all_months = itertools.ifilter(lambda x: x.startswith(pre), keys)
    all_months = sorted((x.split('/')[-1] for x in all_months))
    # This is a bit wasteful on reading in months.  Months 13 .. -13 will be
    # read twice, once for their year and once for the next.
    # Read time is less than a second though, so I'm ok with it.

    # TEMPORARILY Just filter to months past a certain date
    # once I know what vars to use when, filter on year/month and
    # use a dict for multiple dispacth
    #for month in all_months[:-12]:
    all_months_ = itertools.ifilter(lambda x: int(x[1:5]) >= 1999, all_months)
    for month in list(all_months_)[:-12]:
        try:
            df_earn = get_earnings_panel(month, store=store)
            # Write out dfs before makeing panel.
            df_earn.to_hdf(earn_store, month)
            print("Added {}".format(month))
        except KeyError:
            print("Skipping {}".format(month))

    store.close()


#-----------------------------------------------------------------------------
# -- misc --

def plot_changes(earn_store):
    fig, ax = plt.subplots()

    for k, _ in earn_store.iteritems():
        wp = earn_store.select(k)
        if len(wp.A) == 0:
            print("Empty panel for {}".format(k))
            continue
        elif (wp.A.dtypes == object).all():
            print("Dtype issues panel for {}".format(k))
            continue

        try:
            diff = (wp.A - wp.B)['PRERNWA']
        except KeyError:
            diff = (wp.A - wp.B)['PTERNWA']
        try:
            diff.plot(kind='kde', ax=ax)
            print("Added {}".format(k))
        except TypeError as e:
            print('Skipping {} due to'.format(k, e))

    return fig, ax

if __name__ == '__main__':
    main()
