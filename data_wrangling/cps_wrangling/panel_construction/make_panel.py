"""
Take the HDFStore of monthly cps files and construct the pseudo-panel.

Match based on identifiers and (demographic, etc.).
Earnings questions are asked in MIS 4 and 8 only, so about a quarter
of the full sample should be included in the panel (ideally).

Since households cycle out of the survey, if, e.g., MIS 4 is in
January 2012, MIS 8 will be January 2013.  I.e. we get earnings
in the same month of two different years.
"""
from __future__ import division

import itertools
import json
from time import strftime, strptime, struct_time

import arrow
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


def match_naive(df1, df2):
    """
    DataFrame -> DataFrame -> Index
    """
    m1 = df1[df1.HRMIS == 4]
    m2 = df2[df2.HRMIS == 8]
    m = m1.join(m2, how='inner', lsuffix='1', rsuffix='2')
    return m


def smart_match(df1, df2):
    """
    Criteria for a match (ideal):

        1. Indicies match (HRHHID, HRHHID2, PULINENO)
        2. Race is the same
        3. Age within +3 or -1 years?

    Secondry criteria: What if linno is NaN?
    """
    demo = ['PTDTRACE1', 'PTDTRACE2', 'PRTAGE1', 'PRTAGE2', 'PESEX1', 'PESEX2']
    joined = m1.join(m2, how='inner', lsuffix='1', rsuffix='2')
    q = 'PTDTRACE1 != PTDTRACE2 | -1 <= PRTAGE2 - PRTAGE1 <= 3'
    return joined.query(q)


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


def make_full_panel(cps_store, start_month):
    """
    store -> str -> Panel

    start month should look like `m2012_08`

    The full panel for a single wave's time in CPS.  The store layout will be:

        store.month

    where month corresponds to the wave whose first month in
    the survey equals one (MIS=IM=1).  Each node will be a dataframe
    where

                                        data
        (interview_date, unique_id)

    alternatively a panel where the items are interview_date / MIS.
    The unique_id section of the index should be constant through
    the full 8 interview months, modulo survey attrition.
    """
    # TODO: Handle truncated panels (last 7 months) (may be ok).
    pre = '/monthly/data/'
    df1 = cps_store.select(pre + start_month)
    df1 = df1[df1['HRMIS'] == 1]
    idx = df1.index
    keys = cps_store.keys()

    start_ar = arrow.get(start_month, 'mYY_MM')
    rng = [1, 2, 3, 9, 10, 11, 12]
    ars = (pre + start_ar.replace(months=x).strftime('m%Y_%m') for x in rng)
    dfs = (cps_store.select(x) for x in ars if x in keys)

    df_dict = {1: df1}
    for i, df in enumerate(dfs, 1):
        df_dict[i + 1] = df[df['HRMIS'] == i + 1]  # doesn't match (idx)

    return pd.Panel(df_dict)


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
