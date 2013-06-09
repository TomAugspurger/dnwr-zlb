"""
We have data dictionaries parsed in an HDFStore.
We have the cps zipped files.

Combine for an HDFStore of CPS tables.

Note on layout:

cps_store/
    monthly/
        dd/
        data/
            jan1989
            feb1989

Want to keep pythonic names so I can't go 2013-01.

See generic_data_dictionary_parser.Parser.get_store_name for info
on which year gets which dd.
"""

import json

import pandas as pd

settings = json.load(open('info.txt'))

dds = pd.HDFStore(settings['store'])
base_path = settings['base_path']

