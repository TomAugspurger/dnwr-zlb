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


def log_merge(df, ltype):
    """
    DataFrame -> String -> Dict
    """
    if ltype == 'panel':
        year = 'HRYEAR4'
        month = 'HRMONTH'
    else:
        year = 'HRYEAR42'
        month = 'HRMONTH2'
    month = (str(int(df[year].dropna().iloc[0])) + '_' +
             str(int(df[month].dropna().iloc[0])))
    length = len(df)
    d = {month: {"type": ltype, "length": length}}
    return d


def match_panel(df1, df2, log=None):
    # TODO: refactor; combine w/ smart_match
    left_idx = df1.index
    df2 = df2.loc[left_idx]  # left merge

    # filter out spurious matches
    # defined as -1 < delta age < 3 OR sex change
    age_diff = df2['PRTAGE'] - df1['PRTAGE']
    age_idx = age_diff[(age_diff > -1) & (age_diff < 3)].index
    sex_idx = df1['PESEX'][(df1['PESEX'] == df2['PESEX'])].index

    df2 = df2.loc[age_idx.intersection(sex_idx)]

    if log:
        merge_output = log_merge(df2, ltype='panel')
        with open(log, 'a') as f:
            json.dump(merge_output, f)
            f.write('\n')

    return df2


def make_full_panel(cps_store, start_month, settings, keys):
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

    if filter is True, it will pass the dataframes off to smart_match
    with the kwargs. Filter each against the initial?

    """
    # TODO: Handle truncated panels (last 7 months) (may be ok).
    # TODO: filter accoriding to id's (idx)
    # TODO: set names of axis.
    # TODO: fillter nonresponses: df = df[~pd.isnull(df).all(1)] ish
    pre = '/monthly/data/'
    df1 = cps_store.select(pre + start_month)
    df1 = df1[df1['HRMIS'] == 1]

    start_ar = arrow.get(start_month, 'mYY_MM')
    rng = [1, 2, 3, 12, 13, 14, 15]
    ars = (pre + start_ar.replace(months=x).strftime('m%Y_%m') for x in rng)
    dfs = (cps_store.select(x) for x in ars if x.split('/')[-1] in keys)

    df_dict = {1: df1}
    for i, df in enumerate(dfs, 1):
        df_dict[i + 1] = match_panel(df1, df[df['HRMIS'] == i + 1],
                                     log=settings['panel_log'])

    return pd.Panel(df_dict)


def get_finished(settings, pat):
    if pat == 'FINISHED-FULL':
        log_path = settings['panel_log']
    elif pat == 'FINISHED-EARN':
        log_path = settings['earn_log']
    else:
        raise ValueError("FINISHED-EARN or FINISHED_EARN only")
    try:
        with open(log_path) as f:
            finished = [x.split(' ')[-1].strip() for x in f if x.startswith(pat)]
    except IOError:
        finished = []
    return finished


def get_months(settings, store, pat):
    # calling store.keys() was so slow
    all_months = itertools.ifilter(lambda x: x.startswith('m'),
                                   dir(store.root.monthly.data))
    finished = get_finished(settings, pat=pat)
    all_months = itertools.ifilter(lambda x: x not in finished, all_months)
    all_months = (x.split('/')[-1] for x in all_months)
    #TODO: fix the 89-93 ones :(
    all_months = sorted(itertools.ifilter(lambda x: int(x[1:5]) >= 1994, all_months))

    return all_months


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

    all_months = get_months(settings, store, pat='FINISHED-FULL')
    for month in all_months:
        try:
            wp = make_full_panel(store, month, settings, keys=all_months)
            wp.to_hdf(panel_store, key=month, format='f')  # month is wave's MIS=1

            with open(settings['panel_log'], 'a') as f:
                f.write('FINISHED-FULL {}\n'.format(month))
            print('FINISHED {}\n'.format(month))
        except Exception as e:
            with open(settings['panel_log'], 'a') as f:
                f.write("FAILED on {0} with exception {1}\n.".format(month, e))

    all_months = get_months(settings, store, pat='FINISHED-EARN')
    earn_store = pd.HDFStore(settings["earn_store_path"])
    pre = '/monthly/data/'
    for month in all_months:
        month_ar = arrow.get(month, 'mYY_MM')
        key = pre + month_ar.replace(years=1).strftime('m%Y_%m')
        try:
            wp = get_earnings_panel(panel_store, month)
            wp.to_hdf(earn_store, key)  # difference from year before.
            print('Finsihed {}'.format(month))
            with open(settings['earn_log'], 'a') as f:
                f.write("FINISHED-EARN {}\n.".format(month))
        except Exception as e:
            with open(settings['earn_log'], 'a') as f:
                f.write("FAILED on {0} with exception {1}\n.".format(month, e))
    store.close()

if __name__ == '__main__':
    main()
