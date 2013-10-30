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

import itertools as it
import json
from time import strftime, strptime, struct_time

import arrow
import pandas as pd

from checker import Checker


def get_next_month(this_month):
    """
    str -> str

    Takes a string in mYYYY_MM format and returns the next month,
    in the same format.
    """
    struct = strptime(this_month, 'm%Y_%m')
    new_year = [struct.tm_year + 1]
    new_struct = struct_time(it.chain(*[new_year, struct[1:]]))
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


def get_earnings_joined(panel_store, month, settings):
    """
    Store -> str -> DataFrame

    Only run *AFTER* building the full panel (below).

    month will be the second of the two outgoing interviews.
    This will give us the differences from a year before.

    DataFrame is suffixed by 1 or 2
    """
    wp = panel_store.select(month).transpose('minor', 'major', 'items')
    df1, df2 = wp[4], wp[8]
    df1 = df1.convert_objects(convert_numeric=True)
    df2 = df2.convert_objects(convert_numeric=True)
    joined = smart_match(df1, df2, settings)
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

    To keep dtypes the Panel's layout is
    items: df.columns
    major: df.index
    minor: [1,2,..8] the MIS.

    When reading for earn_store you'll need to do:
        wp = panel_store.select(month).transpose('minor', 'major', 'items')
    """
    # TODO: Handle truncated panels (last 7 months) (may be ok).
    # TODO: set names of axis.
    df1 = cps_store.select(start_month)
    df1 = df1[df1['HRMIS'] == 1]

    start_ar = arrow.get(start_month, 'mYY_MM')
    rng = [1, 2, 3, 12, 13, 14, 15]
    ars = (start_ar.replace(months=x).strftime('/m%Y_%m') for x in rng)
    dfs = (cps_store.select(x) for x in ars if x in keys)

    df_dict = {1: df1}
    for i, df in enumerate(dfs, 1):
        df_dict[i + 1] = match_panel(df1, df[df['HRMIS'] == i + 1],
                                     log=settings['panel_log'])
    # Lose dtype info here if I just do from dict.
    # to preserve dtypes:
    wp = pd.Panel.from_dict(df_dict, orient='minor')
    return wp


def get_last_log(fpath):
    with open(fpath, 'r') as f:
        log = f.read()
    starts = filter(lambda x: x.startswith('start time'), log.splitlines())
    try:
        last_start = starts[-1]
    except IndexError:
        last_start = None
    last_log = it.dropwhile(lambda x: x != last_start, log.splitlines())
    return last_log


def get_finished(settings, kind):
    """
    The log format is:
    month,  time started,                    time finished
    1989_01,2013-10-28T20:05:52.995437+00:00,2013-10-28T20:06:01.821006+00:00

    This checks the first column for finished months.
    """
    if kind == 'full_panel':
        log_path = settings['make_full_panel_completed']
    elif kind == 'earn':
        log_path = settings['make_earn_completed']
    else:
        raise ValueError("full_panel or earn only")
    try:
        with open(log_path) as f:
            finished = [x.split(',')[0].strip() for x in f]
    except IOError:
        with open(log_path, 'w') as f:
            pass
        finished = []
    return finished


def get_months(settings, store, kind, skip_finished=True):
    # calling store.keys() was so slow
    all_months = store.keys()
    finished = get_finished(settings, kind=kind)

    if skip_finished:
        return (sorted((x for x in all_months if x.lstrip('m') not in finished)),
                all_months)
    else:
        return (sorted(all_months), all_months)


def write_panel(month, settings, panel_store, cps_store, all_months, start_time):
    """
    Just a wrapper around creation and writing with logging.
    """
    try:
        wp = make_full_panel(cps_store, month, settings, keys=all_months)
        wp.to_hdf(panel_store, key=month, format='f')  # month is wave's MIS=1

        with open(settings['make_full_panel_completed'], 'a') as f:
            f.write('{},{},{}\n'.format(month, start_time, arrow.utcnow()))
        print('FINISHED {}\n'.format(month))
    except Exception as e:
        with open(settings['panel_log'], 'a') as f:
            f.write("FAILED-FULL on {0} with exception {1}\n".format(month, e))


def write_earnings(month, settings, earn_store, panel_store, all_months, start_time):
        """
        Note: If month is 1 then the following dates apply:

            panel_store: m1
            earn_store:  m16   (MIS 3 and 8, wall-time is months 4 and 16)
        """
        month_ar = arrow.get(month, 'mYY_MM')
        key = month_ar.replace(months=15).strftime('m%Y_%m')
        try:
            df = get_earnings_joined(panel_store, month, settings)
            df.to_hdf(earn_store, key)  # difference from year before.
            print('Finsihed {}'.format(month))
            with open(settings['make_earn_completed'], 'a') as f:
                f.write('{},{},{}\n'.format(month, start_time, arrow.utcnow()))
        except Exception as e:
            with open(settings['earn_log'], 'a') as f:
                f.write("FAILED-EARN on {0} with exception {1}\n".format(month, e))
                print("Failed on {} with exception {}.".format(month, e))


def get_touching_months(months, kind='full_panel'):
    """
    [YYYY_MM] -> [YYYY_MM]

    If the cps_store for month m changes, which panels and earning stores
    need to be updated?

    For panels, if month m is changed then we must change months
    m-15
    m-14
    m-13
    m-12
    m-03
    m-02
    m-01
    m
    m+01
    m+02
    m+03
    m+12
    m+13
    m+14
    m+15


    for earn_store its just m+3 and m+15.
    we pass the list since some months will share a common month.
    """
    ars = [arrow.get(month, 'YYYY_MM') for month in months]
    if kind == 'full_panel':
        shifts = [-15, -15, -13, -12, -3, -2, -1, 0, 1, 2, 3, 12, 13, 14, 15]
    elif kind == 'earn':
        shifts = [3, 15]
    else:
        raise ValueError("Just `full_panel` or `earn`.")
    need_update = set(x.replace(months=y) for y in shifts for x in ars)
    for x in need_update:
        yield x.strftime('%Y_%m')


def main():
    """
    Create two panels.
        1. Full Panel. Each node a panel. Key is each wave's first MIS.
            items: [0 .. 8] each wave's MIS.
            major: fields
            minor: micro ID.
        2. Earnings Panel.
    """
    import sys

    try:
        special_months = sys.argv[1]
    except IndexError:
        special_months = False
    with open('settings.txt', 'r') as f:
        settings = json.load(f)

    start_time = arrow.utcnow()
    store_path = settings['store_path']
    store = pd.HDFStore(store_path)
    panel_store = pd.HDFStore(settings['panel_store_path'])

    #---------------------------------------------------------------------------
    # Create Full Panels
    #---------------------------------------------------------------------------
    months_todo, keys = get_months(settings, store, kind='full_panel',
                                   skip_finished=True)
    if special_months:
        with open('update_panels.txt') as f:
            months_todo = get_touching_months([x.rstrip() for x in f])

    print("Panels to create: {}".format(months_todo))
    for month in months_todo:
        write_panel(month, settings, panel_store, store, keys, start_time)

    #---------------------------------------------------------------------------
    # Create Earn DataFrames
    #---------------------------------------------------------------------------
    months_todo, keys = get_months(settings, store, kind='earn',
                                   skip_finished=True)
    if special_months:
        with open('update_panels.txt') as f:
            months_todo = get_touching_months([x.rstrip() for x in f])

    earn_store = pd.HDFStore(settings["earn_store_path"])

    for month in months_todo:
        write_earnings(month, settings, earn_store, panel_store, keys, start_time)

    store.close()
    earn_store.close()
    panel_store.close()

    # clean from update_panels.
    if special_months:
        with open('update_panels.txt', 'w') as f:
            pass

if __name__ == '__main__':
    main()
