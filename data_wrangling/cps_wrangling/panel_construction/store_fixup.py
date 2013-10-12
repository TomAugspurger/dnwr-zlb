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


if __name__ == '__main__':

    settings = json.load(open('settings.txt'))
    raw_path = pathlib.Path(str(settings['raw_monthly_path']))
    dds = pd.HDFStore(settings['store_path'])

    need_fix = get_need_fix(settings)
    for month in need_fix:
        append_to_store(month, settings=settings, skips=[], dds=dds)
