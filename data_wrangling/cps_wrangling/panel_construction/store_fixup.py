from __future__ import division

import itertools as it
import json
import pathlib
import re

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


def main():
    settings = json.load(open('settings.txt'))
    store = pd.HDFStore(settings['store_path'])
    #
    # fix_age_jan2010(settings, store)
    # fix_names(settings, store)
    fix_year_2(settings, store)

if __name__ == '__main__':

    # settings = json.load(open('settings.txt'))
    # raw_path = pathlib.Path(str(settings['raw_monthly_path']))
    # dds = pd.HDFStore(settings['store_path'])

    # need_fix = get_need_fix(settings)
    # for month in need_fix:
    #     append_to_store(month, settings=settings, skips=[], dds=dds)
    main()
