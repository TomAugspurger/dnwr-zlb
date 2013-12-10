from __future__ import (division, print_function, unicode_literals)

from collections import Iterable
import itertools as it
import json

import requests
import pandas as pd
import numpy as np

# Nonfarm Business - Labor Productivity (Output per Hour) - PRS85006093
# Nonfarm Business - Real Hourly Compensation - PRS85006153

#-----------------------------------------------------------------------------
# Globals
URL = "http://api.bls.gov/publicAPI/v1/timeseries/data/"
HEADERS = {'Content-type': 'application/json'}
#-----------------------------------------------------------------------------


def fetch_data(data_l):
    """
    Expects a iterable from make_data. Yields requests.
    """
    if not isinstance(data_l, Iterable):
        data_l = [data_l]
    for data in data_l:
        r = requests.get(URL, data=data, headers=HEADERS)
        if r.status_code == 200:
            yield r
        else:
            raise requests.RequestException(r)


def parse_data(r_l, freq=None):
    """
    Assumes quarterly. May need to factor out.
    """
    dfs = []
    for r in r_l:
        series = r.json()['Results']['series']
        res = []
        for s in series:
            name = s['seriesID']
            data = s['data']
            df = pd.DataFrame(data)
            df = _filter_dates(df)
            # ISO 8601 timestamp
            df['stamp'] = pd.to_datetime(df.year + '-' + df.period, format='%Y-%m-%d')
            df['series_id'] = name
            df = df.set_index(['series_id', 'stamp']).loc[:, 'value']
            df = df.sort_index()
            res.append(df)

        df = pd.concat(res)
        df = df.convert_objects(convert_numeric=True)
        dfs.append(df)

    df = pd.concat(dfs)
    return df.unstack('series_id')


def _filter_dates(df, inplace=True):
    """
    Some series will mix in Annual with Quarterly
    """
    if not inplace:
        df = df.copy()

    if (df.periodName == 'Annual').all():
        df.loc[:, 'period'] = '01-01'
    elif df.periodName.str.match(r'[\dth Quarter, Annual]').all():
        df = df[df.period.str.match('Q0[1,2,3,4]')]  # filter out Q5: annual
        df.loc[:, 'period'] = df['period'].replace({"Q01": "01-01",
                                                    "Q02": "04-01",
                                                    "Q03": "07-01",
                                                    "Q04": "10-01"})
    else:
        raise ValueError("Couldn't infer frequency.")
    return df


def make_data(series, start='2003', end='2011'):
    # BLS limits queries to 10 years because reasons
    n_years = int(end) - int(start)
    if n_years > 10:

        n_queries = int(np.ceil(n_years / 10))
        endpoints = (np.min([int(start) + x * 10, int(end)])
                     for x in np.arange(1, n_queries + 1))
        startpoints = (int(start) + x * 11 for x in np.arange(n_queries))

        for s, e in it.izip(startpoints, endpoints):
            yield json.dumps({"seriesid": series, "startyear": str(s),
                              "endyear": str(e)})
    else:
        yield json.dumps({"seriesid": series, "startyear": start, "endyear": end})


def write_data(r):
    pass


def main():
    data1 = json.dumps({"seriesid": ["PRS85006093", "PRS85006153"],
                        "startyear": "1994", "endyear": "2004"})
    p1 = parse_data(fetch_data(data1))
    data2 = json.dumps({"seriesid": ["PRS85006093", "PRS85006153"],
                       "startyear": "2005", "endyear": "2013"})
    p2 = parse_data(fetch_data(data2))
    df = pd.concat([p1, p2])

    with open('../cps_wrangling/panel_construction/settings.txt', 'r') as f:
        settings = json.load(f)

    analyzed = pd.HDFStore(settings['analyzed_path'])
    df = df.rename(columns={"PRS85006093": "productivity",
                            "PRS85006153": "compensation"})
    df.to_hdf(analyzed, 'bls_productivity_compensation', format='table',
              append=False)

    #--------------------------------------------------------------------------
    # # Industry productivity
    # codes = ["IPUUN8111__L000", "IPUTN722___L000", "IPUKN52211_L000",
    #          "IPUJN5171__L000", "IPUJN511___L000", "IPUIN481___L000",
    #          "IPUHN452___L000", "IPUHN4451__L000", "IPUHN44_45_L000",
    #          "IPUGN42____L000", "IPUEN3361__L000", "IPUEN334___L000",
    #          "IPUCN2211__L000", "IPUBN21____L000"]

    # # only available till 2011
    # data = json.dumps({"seriesid": codes, "startyear": 2003, "endyear": 2011})
    # p2 = parse_data(fetch_data(data))

    #---------------------------------------------------------------------------
    """
    from ftp://ftp.bls.gov/pub/time.series/pr/
    sectors:
        3100 - Durable manufacturing
        3200 - Nondurable manufacturing
        8500 - nonfarm business
    duration - 3 (Index, base year = 100)
    measure:
        09      Labor productivity (output per hour)

    Field #/Data Element    Length          Value(Example)

    1.  series_id           17              PRS30006011

    2.  sector_code         4               3000

    3.  class_code          1               6

    4.  measure_code        2               01

    5.  duration_code       1               1

    6.  seasonal (code)     1               S

    7.  base_year           4               -

    8.  footnote_codes      10              r

    9.  begin_year          4               1988

    10. begin_period        3               Q01

    11. end_year            4               2011

    12. end_period          3               Q03

    The series_id (PRS30006011) can be broken out into:

    Code                                    Value

    survey abbreviation     =               PR
    seasonal (code)         =               S
    sector_code             =               3000
    class_code              =               6
    measure_code            =               01
    duration_code           =               1
    """
    codes = ['PRS31006093', 'PRS32006093', 'PRS85006093']
    df = parse_data(fetch_data(make_data(codes, start='1996', end='2013')))
    # Not returning concated thigns
    renamer = dict(zip(codes, ['durable', 'nondurable', 'business']))
    df = df.rename(columns=renamer)
    df.to_hdf(analyzed, 'major_sectors_output_per_hour', format='t',
              append=False)

    analyzed.close()

if __name__ == '__main__':
    main()
