import json
from datetime import datetime
from dateutil.parser import parse

import pandas as pd
from pandas.io.data import DataReader


def get_names(settings_path='settings.json'):
    """
    For reproducability, don't hardcode the settings

    Parameters
    ----------

    settings_path: str.  path to json file

    Returns
    -------

    store_path: str. path the where the data will be stored.
    store_name: str. name of table in the HDFStore.
    """
    with open(settings_path) as f:
        settings = json.load(f)

    store_path = settings['store_path']
    store_name = settings['store_name']
    return store_path, store_name, settings


def fetch_series(series, settings, cache=True):
    """
    Get the data from FRED's servers.

    Tries to get series from settings, but if none there will grab the
    hardcoded defaults.

    Parameters
    ----------

    settings : dict

    Returns
    -------

    df : DataFrame
    """
    try:
        start, end = map(parse, [settings["start_date"],
                                 settings["end_date"]])
    except KeyError:
        start = datetime(1948, 1, 1)
        end = datetime(2013, 7, 7)

    dfs = [DataReader(ser, data_source="fred", start=start, end=end) for
           ser, _ in series.iteritems()]

    df = pd.concat(dfs, axis=1)
    df = df.rename(columns=series)

    if cache:
        with pd.get_store(settings["store_path"]) as store:
            old = store.select(settings["store_name"])

        df = pd.concat([df, old], axis=1)

    df = df.resample('QS-JAN')  # Start of quarter, begining in January
    return df


def writer(df, settings, store_name=None, store_path=None):
    """
    Write the DataFrame to an HDFStore

    Parameters
    ----------

    df : DataFrame
    settings : dict

    Returns
    -------

    IO
    """
    with pd.get_store(store_path) as store:
        try:
            store.remove(store_name)
        except KeyError:
            pass
        store.append(store_name, df)
        print("Added {0} to {1}".format(store_name, store_path))


def check_cache(settings):
    """
    See which series are already in the store.
    """
    store_path = settings['store_path']

    with pd.get_store(store_path) as store:
        try:
            cols = store.select(settings['store_name']).columns
        except KeyError:
            cols = {}

    series = settings['fred_series']
    new = set(series.values()).difference(cols)
    new = {k: v for k, v in series.iteritems() if v in new}
    return new


def main(cache=True):
    store_path, store_name, settings = get_names()
    if cache:
        series = check_cache(settings)
    else:
        series = settings["fred_series"]

    df = fetch_series(series, settings, cache=cache)
    writer(df, settings, store_name=store_name, store_path=store_path)


if __name__ == '__main__':
    from sys import argv
    try:
        cache = argv[1]
        if cache == "False":
            cache = False
        else:
            cache = True
    except IndexError:
        cache = True
    main(cache=cache)
