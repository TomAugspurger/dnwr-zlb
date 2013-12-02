import json

import requests
import pandas as pd

# Nonfarm Business - Labor Productivity (Output per Hour) - PRS85006093
# Nonfarm Business - Real Hourly Compensation - PRS85006153

#-----------------------------------------------------------------------------
# Globals
URL = "http://api.bls.gov/publicAPI/v1/timeseries/data/"
HEADERS = {'Content-type': 'application/json'}
#-----------------------------------------------------------------------------


def fetch_data(data):
    r = requests.get(URL, data=data, headers=HEADERS)
    if r.status_code == 200:
        return r
    else:
        raise requests.RequestException(r)


def parse_data(r):
    """
    Assumes quarterly. May need to factor out.
    """
    series = r.json()['Results']['series']
    res = []
    for s in series:
        name = s['seriesID']
        data = s['data']
        df = pd.DataFrame(data)
        df = df[df.period.str.match('Q0[1,2,3,4]')]  # filter out Q5: annual
        df['period'] = df['period'].replace({"Q01": "01-01",
                                             "Q02": "04-01",
                                             "Q03": "07-01",
                                             "Q04": "10-01"})
        # ISO 8601 timestamp
        df['stamp'] = pd.to_datetime(df.year + '-' + df.period, format='%Y-%m-%d')
        df['series_id'] = name
        df = df.set_index(['series_id', 'stamp']).loc[:, 'value']
        df = df.sort_index()
        res.append(df)

    df = pd.concat(res)
    df = df.convert_objects(convert_numeric=True)
    return df.unstack('series_id')


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
    df.to_hdf(analyzed, 'bls_productivity_compensation', format='table', append=False)


if __name__ == '__main__':
    main()
