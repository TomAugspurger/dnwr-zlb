"""
Once you have a panel, use this to convert to long form.

Each quarter gets its own node in the `analyzed` HDFStore.
"""
import json
import os

import arrow
import pathlib
import pandas as pd

from data_wrangling.cps_wrangling.analysis.helpers import (
    chunk_quarters, date_parser, read_to_long, make_chunk_name)

from data_wrangling.cps_wrangling.panel_construction.hdf_wrapper import HDFHandler

def make_to_long(panel_h, settings, start=None, stop=None):
    """
    Let's chunk by quarters.
    """
    keys = sorted(panel_h.stores.keys())

    m0 = start or keys[0]
    m0 = date_parser(m0)

    mn = stop or keys[-1]
    mn = date_parser(mn)

    months = [x.strftime('%Y_%m') for x in arrow.Arrow.range('month', m0, mn)
              if x.strftime('m%Y_%m') in keys]

    # Getting some memory pressure. break into chunks, write each out.
    # read proccessed chucnks.
    # Chunking by quarter

    month_chunks = chunk_quarters(months, 3)
    month_chunks = [x for x in month_chunks if len(x) > 0]
    p = pathlib.Path(str(settings['base_path']))
    out_store = HDFHandler(str(p), kind='long', months=month_chunks,
                           frequency='Q')

    for chunk in month_chunks:
        # need the three month chunks... maybe zip up with out_stoure.
        # may need another dict.
        df = read_to_long(panel_h, chunk)
        name = make_chunk_name(chunk)

        # out_store.write(df, name, format='table', append=False)
        s = out_store.stores[name]
        df.to_hdf(s, name, format='table', append=False)
        print("Finished " + str(chunk))


def main():
    with open(os.path.join(os.pardir, 'panel_construction', 'settings.txt'), 'rt') as f:
        settings = json.load(f)

    p = pathlib.Path(str(settings['base_path'])).join('full_panel')
    panel_h = HDFHandler.from_directory(str(p), kind='full_panel')
    make_to_long(panel_h, settings, start='1996_01', stop='2013_06')

if __name__ == '__main__':
    main()