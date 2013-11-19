"""
A class to help handle filesystem stuff with HDFStores.

We want to keep thing broken into separate files so that
a failure (e.g. segfault) doesn't corrept the entire
HDFStore.
"""
import os

import pandas as pd


class HDFHandler(object):
    """
    Useful features:

        - handle batch file creation and naming
        - reading and writing to said files


    Each store type (`full cps`, `panel`, `earnings?`, `analyzed`)
    will be broken up by frequency, either quarterly or monthly.

    Names will be:
    /path/to/base_pat/
        # monthly
        /m1994_01.h5
        /m1994_02.h5

        # quarterly
        /q1994_1.h5
        /q1994_4.h5
    """
    # TODO: context manager
    def __init__(self, settings, kind, months, frequency):
        """
        settings : dict
            must at least contain `base_path`
        kind : str
            one of `full_cps`, `panel`, `earnigns`, `analyzed`.
        months : list
            of months to create. Will overwrite existing ones.
        frequency: str
            one of `monthly` (m), `quarterly` (q)
        """

        self.settings = settings
        self.kind = kind
        self.base_path = settings['base_path']
        self.months = months
        self.frequency = frequency

    def __call__(self, kind, months=None):
        # should return a success/failure code.
        # can choose specific months by passing them here.
        if months is None:
            months = self.months
        if kind == 'panel':
            return self._make_panel(self, months)
        elif kind == 'full_cps':
            return self._make_panel(self, months)
        else:
            raise ValueError

    def _make_panel(self, months):
        """
        Make the files and write months to those files.
        """
        self.stores = self._make_stores(self, months, self.frequency)

    def _make_stores(self, months, frequency):
        if frequency in ('monthly', 'M', 'm'):
            pre = 'm'
        elif frequency in ('quarterly', 'Q', 'q'):
            pre = 'q'
        else:
            raise ValueError("Frequency expected `M` or `Q`")
        base_names = (os.path.join(self.base_path, self.kind, pre + month + '.h5')
                      for month in months)
        stores = []
        for name in base_names:
            stores.append(pd.HDFStore(name))
        return stores
