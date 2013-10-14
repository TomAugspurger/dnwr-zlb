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


def main():
    settings = json.load(open('settings.txt'))
    store = pd.HDFStore(settings['store_path'])
    #
    fix_age_jan2010(settings, store)

if __name__ == '__main__':

    # settings = json.load(open('settings.txt'))
    # raw_path = pathlib.Path(str(settings['raw_monthly_path']))
    # dds = pd.HDFStore(settings['store_path'])

    # need_fix = get_need_fix(settings)
    # for month in need_fix:
    #     append_to_store(month, settings=settings, skips=[], dds=dds)
    main()
