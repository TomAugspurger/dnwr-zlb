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


def smart_match(df1, df2, settings, match_type='age_race', log=False):
    """
    Criteria for a match (ideal):

        1. Indicies match (HRHHID, HRHHID2, PULINENO)
        2. Race is the same
        3. Age within +3 or -1 years?

    age races is one of ['race', 'age', 'age_race', 'age_race_sex', 'sex']
    """
    # TODO: Log various match sizes (naive, age, years, etc)
    # demo = ['PTDTRACE1', 'PTDTRACE2', 'PRTAGE1', 'PRTAGE2', 'PESEX1', 'PESEX2']
    joined = df1.join(df2, how='inner', lsuffix='1', rsuffix='2')

    age = "-1 <= PRTAGE2 - PRTAGE1 <= 3"
    race = "PTDTRACE1 == PTDTRACE2"
    sex = "PESEX1 == PESEX2"
    age_race = ' & '.join([age, race])
    age_race_sex = ' & '.join([age, race, sex])

    crit_d = {'age_race': age_race, 'age_race_sex': age_race_sex,
              'age': age, 'race': race, 'sex': sex}

    def log_merge(df, ltype):
        """
        DataFrame -> String -> Dict
        """
        month = (str(int(df['HRYEAR42'].dropna().iloc[0])) + '_' +
                 str(int(df['HRMONTH2'].dropna().iloc[0])))
        length = len(df)
        d = {month: [{"type": ltype}, {"length": length}]}
        return d

    if log:
        with open(settings['merge_log'], 'a') as f:
            for criteria in crit_d:
                to_log = log_merge(joined.query(crit_d[criteria]), criteria)
                json.dump(to_log, f)

    return joined.query(crit_d[match_type])


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


def get_earnings_panel(panel_store, month):
    """
    Store -> str -> DataFrame

    Only run *AFTER* building the full panel (below).

    month will be the second of the two outgoing interviews.
    This will give us the differences from a year before.

    DataFrame is suffixed by 1 or 2
    """
    wp = panel_store.select(month)
    df1, df2 = wp[4], wp[8]
    joined = smart_match(df1, df2)
    return joined


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
    # TODO: filter accoriding to id's (idx)
    # TODO: set names of axis.
    # TODO: fillter nonresponses: df = df[~pd.isnull(df).all(1)] ish
    pre = '/monthly/data/'
    df1 = cps_store.select(pre + start_month)
    df1 = df1[df1['HRMIS'] == 1]
    idx = df1.index
    keys = cps_store.keys()

    start_ar = arrow.get(start_month, 'mYY_MM')
    rng = [1, 2, 3, 12, 13, 14, 15]
    ars = (pre + start_ar.replace(months=x).strftime('m%Y_%m') for x in rng)
    dfs = (cps_store.select(x) for x in ars if x in keys)

    df_dict = {1: df1}
    for i, df in enumerate(dfs, 1):
        df_dict[i + 1] = df[df['HRMIS'] == i + 1]  # doesn't match (idx)

    return pd.Panel(df_dict)


def get_finished(settings):
    with open(settings['panel_log']) as f:
        finished = [x.split(' ')[-1].strip() for x in f if x.startswith('FINISHED')]

    return finished


def main():
    """
    Create two panels.
        1. Full Panel. Each node a panel. Key is each wave's first MIS.
            items: [0 .. 8] each wave's MIS.
            major: fields
            minor: micro ID.
        2. Earnings Panel.
    """
    with open('settings.txt', 'r') as f:
        settings = json.load(f)

    store_path = settings['store_path']
    store = pd.HDFStore(store_path)
    panel_path = settings['panel_path']
    panel_store = pd.HDFStore(panel_path)
    pre = '/monthly/data/'  # Path inside HDFStore.  Concat to month

    keys = store.keys()
    finished = get_finished(settings)

    all_months = itertools.ifilter(lambda x: x.startswith(pre), keys)
    all_months = itertools.ifilter(lambda x: x not in finished, all_months)
    all_months = sorted((x.split('/')[-1] for x in all_months))

    for month in all_months:
        wp = make_full_panel(store, month)
        wp.to_hdf(panel_store, key=month, format='f')  # month is wave's MIS=1

        with open('panel_log.txt', 'a') as f:
            f.write('FINISHED {}'.format(month))
        print('FINISHED {}\n'.format(month))
    # for month in all_months:
    #     start_ar = arrow.get(month, 'mYY_MM')
    #     # store as second year (so difference).
    #     store_key = start_ar.replace(years=+1).strftime('e%Y_%m')
    #     wpe = get_earnings_panel(start_ar)
    #     wpe.to_hdf(panel_store, key=store_key, format='f')

    store.close()

if __name__ == '__main__':
    main()
