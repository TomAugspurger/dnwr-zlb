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
    def __init__(self, settings, kind, months=None, frequency=None):
        """
        settings : dict
            must at least contain `base_path`
        kind : str
            one of `full_cps`, `panel`, `earnigns`, `analyzed`.
        months : list
            of months to create. Will overwrite existing ones.
            If months is None then just selects all of that kind.
        frequency: str
            one of `monthly` (m), `quarterly` (q)
        """

        self.settings = settings
        self.kind = kind
        self.base_path = settings['base_path']
        self.frequency = frequency
        self.months = months
        if months is None:
            self.select_all(kind)
        else:
            self.stores = self._make_stores(months, frequency)

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

    def select_all(self, kind):
        """
        Called at init when months is None.

        Parameters
        ----------
        kind: str

        Returns
        -------

        list of HDFStores
            appends that list to self.stores
        """
        pass

    def _make_panel(self, months):
        """
        Make the files and write months to those files.
        """
        self.stores = self._make_stores(self, months, self.frequency)

    def _make_stores(self, months, frequency):
        """
        Handle the creation of each HDFStores one for each bin (according
        to frequency).

        months: possibly will remove. just here for overrides.
        freqency: str
            one of `Q` or `M` or synonyms

        Returns
        -------

        list of pd.HDFStores
            files are base_path + kind + freq + month + .h5
        """
        if frequency in ('monthly', 'M', 'm'):
            pre = 'm'
        elif frequency in ('quarterly', 'Q', 'q'):
            pre = 'q'
        else:
            raise ValueError("Frequency expected `M` or `Q`")
        base_names = (os.path.join(self.base_path, self.kind, pre + month + '.h5')
                      for month in months)

        if not os.path.exists(self.base_path):
            os.mkdir(self.base_path)

        if not os.path.exists(os.path.join(self.base_path, self.kind)):
            os.mkdir(os.path.join(self.base_path, self.kind))

        stores = []
        for name in base_names:
            stores.append(pd.HDFStore(name))
        return stores

    def close(self):
        """
        Close every store in self.stores.

        Assumes that self.stores has been set?
        """
        for f in self.stores:
            f.close()

