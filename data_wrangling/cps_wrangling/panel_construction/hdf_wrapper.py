"""
A class to help handle filesystem stuff with HDFStores.

We want to keep thing broken into separate files so that
a failure (e.g. segfault) doesn't corrept the entire
HDFStore.
"""
from itertools import izip
import os

import pathlib
import pandas as pd

from data_wrangling.cps_wrangling.analysis.helpers import date_parser, make_chunk_name


class HDFHandler(object):
    """
    Useful features:

        - handle batch file creation and naming
        - reading and writing to said files


    Each store type (`full cps`, `panel`, `earnings?`, `analyzed`)
    will be broken up by frequency, either quarterly or monthly.

    Names will be:
    /path/to/base_path/
        # monthly
        /kind/m1994_01.h5
        /kind/m1994_02.h5

        # quarterly
        /kind/q1994_1.h5
        /kind/q1994_4.h5
    """
    # TODO: context manager
    def __init__(self, base_path, kind=None, months=None, frequency=None):
        """
        settings : dict
            must at least contain `base_path`
        kind : str
            one of `full_cps`, `panel`, `earnigns`, `analyzed`, 'long'.
        months : list
            of months to create. Will overwrite existing ones.
            If months is None then just selects all of that kind.
        frequency: str
            one of `monthly` (m), `quarterly` (q)
        """
        self.base_path = base_path
        self.kind = kind
        self.frequency = frequency

        if frequency in ('monthly', 'M', 'm'):
            self.pre = 'm'
        elif frequency in ('quarterly', 'Q', 'q'):
            self.pre = 'q'
        elif months is not None:
            raise ValueError("Frequency expected `M` or `Q` if months is not None.")
        else:
            self.pre = frequency

        self.months = months
        if months:
            self.stores = self._select_stores()

    @classmethod
    def from_directory(cls, directory, kind):
        """
        Alternative constructor. Should only be used for collecting
        previously constructed wrappers. Directory is a path (str).
        Returns all stores in that direcotry.
        """
        p = pathlib.Path(directory)
        stores = dict((str(c.name), pd.HDFStore(str(c))) for c in p)
        klass = cls(directory, kind=kind, frequency=kind)
        klass.stores = stores
        return klass

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

    def __getitem__(self, key):
        # TODO: handle ranges
        key = self._sanitize_key(key)
        return self.stores[key]

    def __repr__(self):
        return "A {} container of {} stores".format(self.__class__,
                                                    len(self.stores))

    def __iter__(self):
        return iter(sorted(self.stores.keys()))

    def iteritems(self):
        """
        Returns tuples containing the store/self key and the
        pandas object in the associated store.
        """
        for key in self:
            try:
                yield key, self[key][key]
            except KeyError:
                yield key, None

    def _select_stores(self):
        pre = self.pre
        months = self.months

        if not os.path.exists(self.base_path):
            os.mkdir(self.base_path)

        with_kind = os.path.join(self.base_path, self.kind)
        if not os.path.exists(with_kind):
            os.mkdir(with_kind)

        stores = {}
        if months is None:
            base_names = (os.path.join(with_kind, f) for f in os.listdir(with_kind)
                          if f.endswith('.h5'))
        else:
            if isinstance(months[0], list):
                base_names = (os.path.join(self.base_path, self.kind,
                              make_chunk_name(chunk)) + '.h5'
                              for chunk in months)
            else:
                base_names = (os.path.join(self.base_path, self.kind,
                              pre + month + '.h5') for month in months)
        for name in base_names:
            k, _ = os.path.splitext(os.path.basename(name))
            stores[k] = pd.HDFStore(name)
        return stores

    def _make_panel(self, months):
        """
        Make the files and write months to those files.
        """
        self.stores = self._make_stores(self, months, self.frequency)

    def close(self):
        """
        Close every store in self.stores.

        Assumes that self.stores has been set?
        """
        for f in self.stores:
            self[f].close()

    def write(self, frame, key, *args, **kwargs):
        """
        Similar API to pd.NDFrame.to_hdf()

        Parameters
        ----------

        frame : a pandas object with to_hdf
        key : key to use in the store
        """
        #TODO: need to seprate store key from self key.
        store = self[key]
        frame.to_hdf(store, key, *args, **kwargs)

    def _sanitize_key(self, key):
        if self.kind == 'long':
            return key
        else:
            return self.pre + date_parser(key).strftime('%Y_%m')

    def select(self, key):
        """
        same as store.select()
        """
        key = self._sanitize_key(key)
        return self[key].select(key)

    def apply(self, func, groupby=None, level=None, selector=None,
              *args, **kwargs):
        """
        Apply a function to each pandas object in self. Optionally group by
        groupby, or levels.


        Parameters
        ----------

        - func : callable
        - groupby : str or list of
            columns in each frame to groupby
        - levels : str or list of
            index names to group on
        - *args and **kwargs are passed along to func

        Returns
        -------

        NDFrame
            DataFrame if the result of func is singleton, Panel otherwise.


        """
        # results = []

        aggfuncs = ['mean', 'median', 'mode', 'count', 'sum', 'size']

        def apply_(func, selector):
            for key, df in self.iteritems():
                if selector is None:
                    selector = df.columns
                if groupby:
                    g = df.groupby(groupby)
                elif level:
                    g = df.groupby(level=level)
                else:
                    raise NotImplementedError
                if func in aggfuncs:
                    res = g[selector].agg(func)
                else:
                    res = g[selector].apply(func)
                yield res

        return pd.concat(list(apply_(func, selector, *args, **kwargs)))
