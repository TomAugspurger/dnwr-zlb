import json
from datetime import datetime
from dateutil.parser import parse

import pandas as pd
from pandas.io.data import DataReader


def get_names(settings_path='fred_nipa_settings.json'):
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


def fetch_series(settings):
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
        series = settings["fred_series"]
    except KeyError:
        series = {"GDPC1": "real_gdp",
                  "GDP": "gdp",
                  "PCECC96": "real_pce",
                  "GDPDEF": "gdp_deflator",
                  "PCECTPI": "pce_chain",
                  "PCNDGC96": "real_pce_nondurable",
                  "PCDG": "pce_durable",
                  }
    try:
        start, end = map(parse, [settings["start_date"],
                                 settings["end_date"]])
    except KeyError:
        start = datetime(1947, 1, 1)
        end = datetime(2013, 7, 7)

    dfs = [DataReader(ser, data_source="fred", start=start, end=end) for
           ser, _ in series.iteritems()]

    df = pd.concat(dfs, axis=1)
    df = df.rename(columns=series)
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


def main():
    store_path, store_name, settings = get_names()
    df = fetch_series(settings)
    writer(df, settings, store_name=store_name, store_path=store_path)


if __name__ == '__main__':
    main()
