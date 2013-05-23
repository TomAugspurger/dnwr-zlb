"""


"""
from datetime import datetime

import pandas as pd

def strip_header(f):
    """
    The NIPA files have some meta data at the top.  This function
    strips that data and returns a parsed version.
    """
    name = f.readline()
    unit = f.readline()
    source = f.readline()
    release = f.readline()
    return [name, unit, source, release]  # just for now


def get_index(f):
    """
    After stripping, the next column should be a line of years.
    Followed by a line of quarters (I, .. IV).

    Since they're sadists, they throw some unicode characters in.
    I'm going to grab the first, last, and len and build an index off
    of that.
    """
    years = [x.strip('"\r\n') for x in f.readline().split(',')]
    quarters = [x.strip('"\r\n') for x in f.readline().split(',')]
    y0, q0 = years[2], quarters[2]  # 1 is LINE, 2 is unicode
    yn, qn = years[-1], quarters[-1]
    #
    quarter_dict = {'I': 1,
                    'II': 4,
                    'III': 7,
                    'IV': 10}
    t0 = datetime(int(y0), quarter_dict[q0], 1)
    tn = datetime(int(yn), quarter_dict[qn] + 3, 1)  # I think freq cuts a side
    return pd.DatetimeIndex(start=t0, end=tn, freq='Q-JAN')


def get_data(f, index, return_cols=None):
    """
    Once the header and indecies have been stripped.
    """
    df = pd.read_csv(f, header=None, index_col=[1], na_values='---', skip_footer=7)
    df = df[df.columns[1:]]  # Get rid of the outer index
    df.columns = index
    df = df.T
    # re.sub(r'\(|\)| | -', '_', a.strip(' '))
    df.columns = [x.strip(' ').replace(' ', '_').lower() for x in df.columns]
    if return_cols is None:
        return_cols = df.columns
    return df[return_cols]


def main(f):
    meta = strip_header(f)
    idx = get_index(f)
    df = get_data(f, idx)
    return df

def run():
    f = open('/Volumes/HDD/Users/tom/DataStorage/nipa/nipa_gdp_1947-2013.csv')
    strip_header(f)
    f.close()

if __name__ == '__main__':
    import sys
    dir_ = sys.argv[1]

    for fname in ['nipa_deflator_1947-2013.csv',
                  'nipa_gdp_1947-2013.csv',
                  'nipa_consumption_1947_2013.csv']:
        with open(dir_ + fname) as f:
            df = main(f)
            df.to_csv(dir_ + 'cleaned_' + fname)
