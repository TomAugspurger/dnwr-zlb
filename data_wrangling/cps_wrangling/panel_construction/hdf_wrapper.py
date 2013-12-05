"""
A class to help handle filesystem stuff with HDFStores.

We want to keep thing broken into separate files so that
a failure (e.g. segfault) doesn't corrept the entire
HDFStore.
"""
import os
import shutil
import subprocess

import pathlib
import pandas as pd
from pandas.core.common import is_list_like

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

        # set by groupby later
        self._full_cache = None
        self._full_cache_args = None

    @classmethod
    def from_directory(cls, directory, kind):
        """
        Alternative constructor. Should only be used for collecting
        previously constructed wrappers. Directory is a path (str).
        Returns all stores in that direcotry.
        """
        p = pathlib.Path(directory)
        stores = dict((str(c.name.split('.')[0]), pd.HDFStore(str(c))) for c in p
                      if not c.name.startswith('.'))
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
        return self.stores[key]

    def __repr__(self):
        return "A {} container of {} stores".format(self.__class__,
                                                    len(self.stores))

    def __iter__(self):
        return iter(sorted(self.stores.keys()))

    def keys(self):
        return self.stores.keys()

    def iteritems(self, **kwargs):
        """
        Returns tuples containing the store/self key and the
        pandas object in the associated store.

        kwargs are passed to pandas.HDFStore.select()
        """
        for key in self:
            try:
                yield key, self[key].select(key, **kwargs)
            except KeyError:
                yield key, None

    def select_all(self, **kwargs):
        if self._full_cache_args == kwargs:
            return self._full_cache
        else:
            self._full_cache_args = kwargs
            df = pd.concat((frame for _, frame in self.iteritems(**kwargs)))
            df = df.sort_index()
            self._full_cache = df
            return df

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

    def select(self, key, **kwargs):
        """
        same as store.select()

        kwargs are all passed along as is.
        """
        return self[key].select(key, **kwargs)

    def apply(self, func, groupby=None, level=None, selector=None,
              select_kwargs=None, *args, **kwargs):
        """
        Apply a function to each pandas object in self. Optionally group by
        groupby, or levels.


        Parameters
        ----------

        - func : callable
            optinally list of funcs for agg
        - groupby : str or list of
            columns in each frame to groupby
        - levels : str or list of
            index names to group on
        - selector : str or list of
            subset of columsn to return. Default None
        - select_kwargs : dict
            keyword argument to pass along to HDFStore.select(). Default None
        - *args and **kwargs are passed along to func

        Returns
        -------

        NDFrame
            DataFrame if the result of func is singleton, Panel otherwise.

        Notes
        -----

        Stuff on disk is chunked according to some scheme. In my case it's
        quarterly. So if you groupby some group that spans multiple files,
        (e.g. sex, race, anything else really), your going to *have* to read
        the entire dataset into memory.  I'd like to cache that somewhere
        on self.
        """
        select_kwargs = select_kwargs if select_kwargs else {}
        aggfuncs = ['mean', 'median', 'mode', 'count', 'sum', 'size', 'std']

        if is_list_like(func) and all([f in aggfuncs for f in func]):
            the_attr = 'agg'
        elif is_list_like(func):
            raise TypeError("Multiple functions only work with aggfuncs")
        elif func in aggfuncs:
            the_attr = 'agg'
        else:
            the_attr = 'apply'

        reset = bool(groupby) and bool(level)

        if groupby and not is_list_like(groupby):
            groupby = [groupby]

        if level and not is_list_like(level):
            level = [level]

        if reset:
            groupby.extend(level)

        def apply_(func, selector):
            for key, df in self.iteritems(**select_kwargs):
                if reset:
                    # will set index later back to ids
                    df = df.reset_index()

                if selector is None:
                    selector = df.columns
                if groupby:
                    g = df.groupby(groupby)
                elif level:
                    g = df.groupby(level=level)
                else:
                    raise NotImplementedError
                res = getattr(g[selector], the_attr)(func)
                # if reset:
                #     res = res.set_index(set(ids).intersection(groupby))
                yield res

        if level is None:
            if self._full_cache is None:
                # self should be immuatble... hrff
                self._full_cache = self.select_all(**select_kwargs)
            return getattr(self.full_cache.groupby(groupby)[selector], the_attr)(func)

        else:
            return pd.concat(list(apply_(func, selector, *args, **kwargs)))

    def map(self, func, selector=None):
        """
        Apply a non grouping function to each frame.

        Parameters
        ----------

        func : function :: Frame -> a
        selector : str
            subset of columns to operate on

        Returns
        -------

        [a]

        Examples
        --------

        long.map(pd.value_counts, selector='same_employer')
        # returns a DataFrame of value_counts
        """

        results = []

        for key, df in self.iteritems():
            # need to think about handling methods this way.
            sub = df[selector]
            if hasattr(sub, str(func)):
                res = sub.getattr(func)()
            else:
                res = func(sub)
            results.append(res)

        return pd.concat(results)

    def compress(self):
        """
        Runs ptrepcack over self.stores and closes.
        """

        stores = self.stores
        for _, v in stores.iteritems():
            src = v.filename
            shutil.move(src, src + '_')
            subprocess.call(["ptrepack", "--complevel=9", src + '_', src])
            os.remove(src + '_')

        self.close()
