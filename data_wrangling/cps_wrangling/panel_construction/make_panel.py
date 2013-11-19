"""
Take the HDFStore of monthly cps files and construct the pseudo-panel.

Match based on identifiers and (demographic, etc.).
Earnings questions are asked in MIS 4 and 8 only, so about a quarter
of the full sample should be included in the panel (ideally).

Since households cycle out of the survey, if, e.g., MIS 4 is in
January 2012, MIS 8 will be January 2013.  I.e. we get earnings
in the same month of two different years.


The CPS messed up the IDs for months 1995-06 through 1995-08 (inclusive).
They didn't return to the old schema.
This means that the following panels will have the following holes:

1994_03 x x x x x x x _
1994_04 x x x x x x _ _
1994_05 x x x x x _ _ _
1994_06 x x x x _ _ _ _
1994_07 x x x x _ _ _ _
1994_08 x x x x _ _ _ _
1994_09 x x x x _ _ _ _
1994_10 x x x x _ _ _ _
1994_11 x x x x _ _ _ _
1994_12 x x x x _ _ _ _
1995_01 x x x x _ _ _ _
1995_02 x x x x _ _ _ _
1995_03 x x x _ _ _ _ _
1995_04 x x _ _ _ _ _ _
1995_05 x _ _ _ _ _ _ _
1995_06 _ _ _ _ _ _ _ _
1995_07 _ _ _ _ _ _ _ _
1995_08 _ _ _ _ _ _ _ _

Which means we lose earnings *comparisions* for the following months
(following the earnings method of reporting: k = second obs (MIS=8),
    which is 15 months in the future of the above table).

[ 1995_06 .. 1996_11 ]
"""
from __future__ import division

import itertools as it
import json
from time import strftime, strptime, struct_time

import arrow

import pandas as pd

from data_wrangling.cps_wrangling.analysis import add_to_panel
from hdf_wrapper import HDFHandler

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


def smart_match(df):
    """
    Criteria for a match (ideal):

        1. Indicies match (HRHHID, HRHHID2, PULINENO)
        2. Race is the same
        3. Age within +3 or -1 years?

    age races is one of ['race', 'age', 'age_race', 'age_race_sex', 'sex']
    """
    # TODO: Log various match sizes (naive, age, years, etc)
    diffed = df.xs(4, level='minor') - df.xs(8, level='minor')

    good_age = (diffed['PRTAGE'] >= -1) & (diffed['PRTAGE'] <= 3)
    good_race = diffed.PTDTRACE == 0
    return df.unstack()[good_age & good_race].stack()


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


def huhhnum_to_hrhhid_index(df1, df2):
    """
    When they switched names they also changed the numbering.
    df1 will have the older index values, df2 will have the newer.

    From the HRHHID2 doc:

        Part 1 of this number is found in columns 1-15 of the record.
    Concatenate this item with Part 1 for matching forward in time.

    The component parts of this number are as follows:
    71-72   Numeric component of the sample number (HRSAMPLE)
    73-74   Serial suffix-converted to numerics (HRSERSUF)
    75      Household Number (HUHHNUM)
    """

    ts1 = pd.to_datetime(str(df1.timestamp.dropna().unique()[0]))
    ts2 = pd.to_datetime(str(df2.timestamp.dropna().unique()[0]))

    if ts1 >= pd.datetime(2003, 2, 1) and ts2 <= pd.datetime(2004, 5, 1):
        ids = df2.index.names
        replace = df2.reset_index()
        replace['HRHHID2'] = replace['HRHHID2'] % 10
        replace = replace.set_index(ids)
        return replace
    else:
        return df2


def match_panel(df1, df2, log=None):
    # TODO: refactor; combine w/ smart_match
    df2 = huhhnum_to_hrhhid_index(df1, df2)
    left_idx = df1.index
    df2 = df2.loc[left_idx]  # left merge
    # When the CPS messed up the ids in June 1995.
    if pd.isnull(df2).all().all():
        return None
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
    # start_ar = arrow.get(start_month, 'mYY_MM')
    # rng = [1, 2, 3, 12, 13, 14, 15]
    # ars = (start_ar.replace(months=x).strftime('/m%Y_%m') for x in rng)
    # dfs = (cps_store.select(x) for x in ars if x in keys)

    def gen_dfs(start_month):
        start_ar = arrow.get(start_month, 'YYYY_MM')
        rng = [0, 1, 2, 3, 12, 13, 14, 15]
        ars = (start_ar.replace(months=x).strftime('/m%Y_%m') for x in rng)
        for i, x in enumerate(ars, 1):
            try:
                df = cps_store.select(x)
                yield df[df['HRMIS'] == i]
            except KeyError:
                yield None

    dfs = gen_dfs(start_month)

    df1 = next(dfs)

    ids = ["HRHHID", "HRHHID2", "PULINENO"]
    if not ids == df1.index.names:
        df1.index.names = ids

    df_dict = {1: df1}
    for i, dfn in enumerate(dfs, 2):
        if not dfn.index.names == ids:
            dfn.index.names = ids
        df_dict[i] = match_panel(df1, dfn, log=settings['panel_log'])
    # Lose dtype info here if I just do from dict.
    # to preserve dtypes:
    df_dict = {k: v for k, v in df_dict.iteritems() if v is not None}
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

        with open(settings['make_full_panel_completed'], 'a') as f:
            f.write('{},{},{}\n'.format(month, start_time, arrow.utcnow()))
        print('FINISHED {}\n'.format(month))
    except Exception as e:
        with open(settings['panel_log'], 'a') as f:
            f.write("FAILED-FULL on {0} with exception {1}\n".format(month, e))


def get_earnings(month, settings, earn_store, panel_store, all_months, start_time):
    """
    Note: If month is 1 then the following dates apply:

        panel_store: m1
        earn_store:  m16   (MIS 4 and 8, wall-time is months 4 and 16)
    """
    try:
        wp = panel_store.select(month)
        earn_months = wp.loc[:, :, (4, 8)]
        # Bug in pandas w/ toframe on Multi so I have to T.T.stack()
        df = earn_months.transpose(1, 0, 2).to_frame(
            filter_observations=False).T.stack().convert_objects(convert_numeric=True)

        return smart_match(df)
    except Exception as e:
        with open(settings['earn_log'], 'a') as f:
            f.write("FAILED-EARN on {0} with exception {1}\n".format(month, e))
            print("Failed on {} with exception {}.".format(month, e))
        raise


def write_earnings(df, earn_store, month, settings, start_time):
    month_ar = arrow.get(month, 'mYY_MM')
    key = month_ar.replace(months=15).strftime('m%Y_%m')
    try:
        earn_store.remove(key)
        df.to_hdf(earn_store, key)  # difference from year before.
    except KeyError:
        df.to_hdf(earn_store, key)  # difference from year before.
    print('Finsihed {}'.format(month))
    with open(settings['make_earn_completed'], 'a') as f:
        f.write('{},{},{}\n'.format(month, start_time, arrow.utcnow()))


def get_touching_months(months, kind='full_panel', direct=False):
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

    for earn_store its m+j+3 and m+j+15 for all j above.
    we pass the list since some months will share a common month.

    This may not be strictly necessary. If I update month m in panel P,
    then I need only update earnings for months P_m+3 and P_m+15, i.e.
    not for P_j+3 and P_j+15 forall j in P. But it doesn't take that
    long to do the earnings panels anyway.
    """
    try:
        ars = [arrow.get(month, 'YYYY_MM') for month in months]
    except arrow.parser.ParserError:
        ars = arrow.get(months, 'YYYY_MM')

    if kind == 'full_panel' and direct:
        shifts = [0, 1, 2, 3, 12, 13, 14, 15]
        need_update = (ars.replace(months=y) for y in shifts)

    elif kind == 'earn' and direct:
        raise NotImplementedError
    elif kind == 'full_panel' and not direct:
        shifts = [-15, -15, -13, -12, -3, -2, -1, 0, 1, 2, 3, 12, 13, 14, 15]
        need_update = set(x.replace(months=y) for y in shifts for x in ars)
    elif kind == 'earn' and not direct:
        shifts = [3, 15]
        need_update = set(x.replace(months=y) for y in shifts for x in need_update)
    else:
        raise ValueError("Just `full_panel`, `earn`, or 'direct', for kind. Got {0} "
                         "instead.".format(kind))
    for x in need_update:
        yield x.strftime('/m%Y_%m')


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
    cps_store = pd.HDFStore(store_path)
    panel_store = pd.HDFStore(settings['panel_store_path'])
    earn_store = pd.HDFStore(settings["earn_store_path"])

    #---------------------------------------------------------------------------
    # Create Full Panels
    #---------------------------------------------------------------------------
    months_todo, keys = get_months(settings, cps_store, kind='full_panel',
                                   skip_finished=True)
    if special_months:
        with open('update_panels.txt') as f:
            months_todo = get_touching_months([x.rstrip() for x in f])

    # Just 1994+ for now
    months_todo = [x for x in months_todo if int(x[2:6]) >= 1994]

    print("Panels to create: {}".format(months_todo))

    panel_h = HDFHandler(settings, kind='full_panel', months=months_todo,
                         frequency='monthly')

    for month in months_todo:
        wp = make_full_panel(cps_store, month, settings, keys=keys)
        wp = add_to_panel.add_flows(month.strip('/m'), panel_store, frame=wp)
        wp = add_to_panel.add_history(month.strip('/m'), panel_store, frame=wp)
        panel_h.write(wp, key=month, append=False, format='f')

    #---------------------------------------------------------------------------
    # Create Earn DataFrames
    #---------------------------------------------------------------------------
    months_todo, keys = get_months(settings, cps_store, kind='earn',
                                   skip_finished=True)
    if special_months:
        with open('update_panels.txt') as f:
            months_todo = get_touching_months([x.rstrip() for x in f], kind='earn')

    # Just 1994+ for now
    months_todo = [x for x in months_todo if int(x[2:6]) >= 1994]

    for month in months_todo:
        try:
            df = get_earnings(month, settings, earn_store,
                              panel_store, keys, start_time)
        except:
            print("Failed on {}. Logged elsewhere".format(month))
        write_earnings(df, earn_store, month, settings, start_time)

    cps_store.close()
    earn_store.close()
    panel_store.close()

    # clean from update_panels.
    if special_months:
        df = pd.read_csv(settings['make_full_panel_completed'],
                         names=['month', 'start', 'end'],
                         parse_dates=['start', 'end'])
        finished = df[df['start'] == start_time.datetime]['month']
        finished = finished.str.strip('/m')

        with open('update_panels.txt', 'r+') as f:
            months = [x.rstrip() for x in f]
            unfinished = [y for y in months if y not in finished.values]
            f.seek(0)
            f.write('\n'.join(unfinished))

if __name__ == '__main__':
    main()
