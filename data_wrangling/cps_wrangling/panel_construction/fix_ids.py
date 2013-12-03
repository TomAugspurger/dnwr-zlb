import json
import os

import arrow
import pandas as pd

from data_wrangling.cps_wrangling.panel_construction.make_panel import write_panel
from data_wrangling.cps_wrangling.analysis import add_to_panel


def fix_panel_store(months, panel_store, ids):
    for month in months:
        wp = panel_store.select(month)
        wp.major_axis.names = ids
        wp.to_hdf(panel_store, key=month, append=False, format='f')
        print("Finished " + month)


def full_fix(month, settings, panel_store, cps_store, keys):
    """
    Actually recreates the panel.

    currnetly on:
    In [39]: chunk
    Out[39]: ['2004_04', '2004_05', '2004_06']

    m2003_02 thru m2004_04
    """
    write_panel(month, settings, panel_store, cps_store, keys, arrow.utcnow())
    add_to_panel.add_flows(month.strip('/m'), panel_store)
    add_to_panel.add_history(month.strip('/m'), panel_store)


def main():
    with open(os.path.join(os.pardir, 'panel_construction', 'settings.txt'), 'rt') as f:
        settings = json.load(f)

    panel_store = pd.HDFStore(settings['panel_store_path'])
    cps_store = pd.HDFStore(settings['store_path'])
    m0 = arrow.get('1994-01-01', format='YYYY-MM-DD')
    mn = arrow.get('2013-06-01', format='YYYY-MM-DD')

    keys = cps_store.keys()
    months = (x.strftime('m%Y_%m') for x in arrow.Arrow.range('month', m0, mn)
              if x.strftime('/m%Y_%m') in keys)

    # ids = ["HRHHID", "HRHHID2", "PULINENO"]

    for month in months:
        full_fix(month, settings, panel_store, cps_store, keys)


if __name__ == '__main__':
    main()


