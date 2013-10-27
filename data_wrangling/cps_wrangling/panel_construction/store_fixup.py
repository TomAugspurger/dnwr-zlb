from __future__ import division

import itertools as it
import json
import os
import pathlib
import re

import arrow
import pandas as pd

from make_hdf_store import append_to_store


def get_need_fix(settings):
    store_log = settings["store_log"]
    reg = re.compile(r'FAILED ([\w,/]*?.gz) .*')

    with open(store_log) as f:
        failures = it.ifilter(lambda x: x.startswith("FAILED"), f.readlines())
        tofix = it.ifilter(lambda x: x.endswith('str.\n'), failures)
        paths_tofix = (reg.match(x).groups()[0] for x in tofix)
        pathlibs_to_fix = (pathlib.PosixPath(x) for x in paths_tofix)
        return pathlibs_to_fix


def fix_age_jan2010(settings, store):
    """
    Missed need to change jan2010 PEAGE -> PRTAGE
    """
    all_months = settings["month_to_dd"]
    needfix = it.ifilter(lambda x: all_months[x] == 'jan2010', all_months)
    for month in needfix:
        month = month[:4] + '_' + month[-2:]
        name = '/monthly/data/m' + month
        df = store.select(name)
        df = df.rename(columns={'PEAGE': 'PRTAGE'})
        try:
            store.remove(name)
        except KeyError:
            pass
        store.append(name, df)
        print("Fixed {}".format(month))


def fix_names(settings, store):
    """
    Bad names on the older ones. At leas

    dds:

    names: {'HdYEAR': "HRYEAR4", "HdMONTH": "MHRONTH"}
    """
    all_months = settings["month_to_dd"]
    needfix = it.ifilter(lambda x: all_months[x] in ('jan1989'),
                         all_months)
    for month in needfix:
        month = month[:4] + '_' + month[-2:]
        name = '/monthly/data/m' + month
        df = store.select(name)
        df = df.rename(columns={"HdYEAR": "HRYEAR4", "HdMONTH": "HRMONTH"})
        df["PRTAGE"] = df["AdAGEDG1"] * 10 + df["AdAGEDG2"]
        df = df.drop(['AdAGEDG1', 'AdAGEDG2'], axis=1)
        try:
            store.remove(name)
        except KeyError:
            pass
        store.append(name, df)


def fix_year(settings, store):
    """
    "jan1994", "apr1994", "jun1995", "sep1995"
    """
    all_months = settings["month_to_dd"]
    needfix = it.ifilter(lambda x: all_months[x] in ("jan1994", "apr1994",
                                                     "jun1995", "sep1995"),
                         all_months)
    for month in needfix:
        month = month[:4] + '_' + month[-2:]
        name = '/monthly/data/m' + month
        df = store.select(name)
        df.HRYEAR4 = 1900 + df.HRYEAR
        df = df.drop('HRYEAR', axis=1)
        try:
            store.remove(name)
        except KeyError:
            pass
        store.append(name, df)
        print('added {}'.format(month))


def fix_year_2(settings, store):
    bad = ['m1994_01', 'm1994_02', 'm1994_03', 'm1994_04', 'm1994_05', 'm1994_06',
           'm1994_07', 'm1994_08', 'm1994_09', 'm1994_10', 'm1994_11', 'm1994_12',
           'm1995_01', 'm1995_02', 'm1995_03', 'm1995_04', 'm1995_05', 'm1995_06',
           'm1995_07', 'm1995_08', 'm1995_09', 'm1995_10', 'm1995_11', 'm1995_12',
           'm1996_01', 'm1996_02', 'm1996_03', 'm1996_04', 'm1996_05', 'm1996_06',
           'm1996_07', 'm1996_08', 'm1996_09', 'm1996_10', 'm1996_11', 'm1996_12',
           'm1997_01', 'm1997_02', 'm1997_03', 'm1997_04', 'm1997_05', 'm1997_06',
           'm1997_07', 'm1997_08', 'm1997_09', 'm1997_10', 'm1997_11', 'm1997_12']
    for month in bad:
        name = '/monthly/data/' + month
        yd = int(month[4])
        df = store.select(name)
        df['HRYEAR4'] = 1990 + yd
        try:
            store.remove(name)
        except KeyError:
            pass
        store.append(name, df)
        print(df.HRYEAR4)
        print('added {}'.format(month))


def fix_after_check(settings, store):
    """
    need to have a panel_check.json file.
    """
    panel_store = pd.HDFStore(settings["panel_path"])

    good_cols_by_dd = settings['col_rename_by_dd']
    # just nonempty
    good_cols_by_dd = {k: v for k, v in good_cols_by_dd.iteritems() if v}
    month_to_dd = settings['month_to_dd']
    tofix = it.ifilter(lambda x: month_to_dd[x] in good_cols_by_dd.keys(),
                       month_to_dd)
    for month in tofix:
        name = '/monthly/data/m' + month[:4] + '_' + month[-2:]
        print(name)
        dd = month_to_dd[month]
        good_cols = good_cols_by_dd[dd]
        df = store.select(name)
        df = df.rename(columns=good_cols)
        try:
            store.remove(name)
        except KeyError:
            pass
        store.append(name, df)
        print("Fixed month {0} at {1}".format(month, arrow.utcnow()))

        new_name = name.split('/')[-1]
        try:
            wp = panel_store.select(new_name)
        except KeyError:
            print("No panel for {}".format(month))
            continue

        wp = wp.rename(good_cols, axis='minor')
        try:
            panel_store.remove(new_name)
        except KeyError:
            pass
        wp.to_hdf(panel_store, key=new_name, format='f')  # month is wave's MIS=1
        print("Fixed panel {0} at {1}".format(month, arrow.utcnow()))


def fix_age_older(settings, store):
    """
    Missed need to change jan2010 PEAGE -> PRTAGE
    """
    all_months = settings["month_to_dd"]
    with open('panel_log.txt') as f:
        g = it.dropwhile(lambda x: not x.startswith('start time: 2013-10-21T00:46:47.102951+00:00'), f)
        failed = it.ifilter(lambda x: x.endswith("u'no item named PRTAGE'\n"), g)
        needfix = sorted(it.imap(lambda x: x.split(' ')[2], failed))

    def combine_age(df, dd_name):
        """For jan89 and jan92 they split the age over two fields."""
        df["PRTAGE"] = df["AdAGEDG1"] * 10 + df["AdAGEDG2"]
        return df

    for month in needfix:
        name = '/monthly/data/' + month
        df = store.select(name)
        df = df.rename(columns={'PEAGE': 'PRTAGE'})
        try:
            store.remove(name)
        except KeyError:
            pass
        store.append(name, df)


def fix_age_early_2012(settings, store):
    """
    missed age change for 2012-01 : 2012-04 (inclusive)
    """
    needfix = ['2012_01', '2012_02', '2012_03', '2012_04']
    for month in needfix:
        name = '/monthly/data/m' + month
        df = store.select(name)
        df = df.rename(columns={'PEAGE': 'PRTAGE'})
        try:
            store.remove(name)
        except KeyError:
            pass
        store.append(name, df)

    with open('update_panels.txt', 'w') as f:
        f.write(('\n'.join(needfix)))


def fix_index_on_89_92(settings, store):
    # TODO: have to run this one.
    d = settings['month_to_dd']
    needfix = [x[:4] + '_' + x[4:] for x in d if d[x] in ('jan1989', 'jan1992')]
    for month in needfix:
        name = '/monthly/data/m' + month
        df = store.select(name)
        df = df.rename(columns={'PEAGE': 'PRTAGE'})
        try:
            store.remove(name)
        except KeyError:
            pass
        store.append(name, df)

    with open('update_panels.txt', 'w') as f:
        f.write(('\n'.join(needfix)))


def fix_dd_on_old(settings, store):
    """
    Had the wrong dd from some months. These are the correct ones.

    "cpsb9404": "apr1995",
    "cpsb9405": "apr1995",
    "cpsb9406": "apr1995",
    "cpsb9407": "apr1995",
    "cpsb9408": "apr1995",
    "cpsb9409": "apr1995",
    "cpsb9410": "apr1995",
    "cpsb9411": "apr1995",
    "cpsb9412": "apr1995",
    "cpsb9501": "apr1995",
    "cpsb9502": "apr1995",
    "cpsb9503": "apr1995",
    "cpsb9504": "apr1995",
    "cpsb9505": "apr1995",
    "cpsb9506": "jun1995",
    "cpsb9507": "jun1995",
    """
    needfix = ["cpsb9404", "cpsb9405", "cpsb9406", "cpsb9407", "cpsb9408",
               "cpsb9409", "cpsb9410", "cpsb9411", "cpsb9412", "cpsb9501",
               "cpsb9502", "cpsb9503", "cpsb9504", "cpsb9505", "cpsb9506",
               "cpsb9507", "cpsb9508"]
    p = pathlib.PosixPath(str(settings['raw_monthly_path']))
    files = list(iter(p))
    posix_fix = [x for x in files if str(x.parts[-1]).rstrip('.gz') in needfix]
    dds = pd.HDFStore(settings['store_path'])
    for month in posix_fix:
        append_to_store(month, settings, skips=[], dds=dds, skip=False)

    needfix = [x[:4] + '_' + x[4:] for x in needfix]

    with open('update_panels.txt', 'a') as f:
        f.write(('\n'.join(needfix)))
        f.write('\n')


def wrong_race_name(settings, store):
    """
    NOT FIXED.  I have no idea where this one came from. The panels under
    months: m2001_10 m2001_11 m2001_12 m2002_01 m2002_02 m2002_03 m2002_04
            m2002_05 m2002_06 m2002_07 m2002_08 m2002_09 m2002_10 m2002_11
            m2002_12

    had race under PRDTRACE when it should be under PTDTRACE.
    The actual messed up months are 2003-01
    """
    months = ["2001_10",
              "2001_11",
              "2001_12",
              "2002_01",
              "2002_02",
              "2002_03",
              "2002_04",
              "2002_05",
              "2002_06",
              "2002_07",
              "2002_08",
              "2002_09",
              "2002_10",
              "2002_11",
              "2002_12"]
    GOOD = 'PTDTRACE'
    BAD = 'PRDTRACE'
    for month in months:
        name = '/m' + month
        wp = store.select(name)
        wp2 = wp.drop(GOOD, axis=2)
        wp2.rename(minor_axis={BAD: GOOD}, inplace=True)
        wp.update(wp2)
        try:
            store.remove(name)
        except KeyError:
            pass
        wp.to_hdf(store, key=name, format='f')
        print('finsished {}'.format(month))

    with open('update_panels.txt', 'w') as f:
        f.write(('\n'.join(months)))


def main():
    settings = json.load(open('settings.txt'))
    # store = pd.HDFStore(settings['store_path'])
    store = pd.HDFStore(settings['panel_store_path'])
    #
    # fix_age_jan2010(settings, store)
    # fix_names(settings, store)
    # fix_year_2(settings, store)
    # fix_after_check(settings, store)
    # fix_age_older(settings, store)
    # fix_age_jan2010(settings, store)
    "fix_age_early_2012(settings, store)"
    # fix_dd_on_old(settings, store)
    wrong_race_name(settings, store)

if __name__ == '__main__':

    # settings = json.load(open('settings.txt'))
    # raw_path = pathlib.Path(str(settings['raw_monthly_path']))
    # dds = pd.HDFStore(settings['store_path'])

    # need_fix = get_need_fix(settings)
    # for month in need_fix:
    #     append_to_store(month, settings=settings, skips=[], dds=dds)
    main()
