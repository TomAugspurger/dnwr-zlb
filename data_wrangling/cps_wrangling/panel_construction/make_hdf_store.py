"""
We have data dictionaries parsed in an HDFStore.
We have the cps zipped repo.

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

They claim to use
    (HHID, HHNUM, LINENO)
for '94 that is "HRHHID", "HUHHNUM", "PULINENO"
and validate with
    sex, age, race


Possiblye interested in

    PTERNH1C-Earnings-hourly pay rate,excluding overtime
    PTERNH2-T Earnings-(main job)hourly pay rate,amount
**  PTWK-T Earnings-weekly-top code flag  **

"""
import os
import json
import subprocess

import pandas as pd


def tst_setup(n=10):
    settings = json.load(open('info.txt'))
    dds = pd.HDFStore(settings['store_path'])
    base_path = settings['base_path']
    repo_path = settings['repo_path']
    dd = dds.select('/monthly/dd/jan1998')
    pth = '/Volumes/HDD/Users/tom/DataStorage/CPS/monthly/cpsb9810'
    widths = dd.length.tolist()
    df = pd.read_fwf(pth, widths=widths, nrows=n, names=dd.id.values)
    return df, dd, dds


def runner(fname, n=10, settings=json.load(open('info.txt'))):
    dds = pd.HDFStore(settings['store_path'])
    no_ext = ''.join(fname.split('.')[:-1])
    dd_month = settings['month_to_dd_by_filename'][no_ext.split('/')[-1] + '.Z']
    dd = dds.select('/monthly/dd/' + dd_month)
    widths = dd.length.tolist()
    if fname.endswith('.Z'): 
        pth = ''.join(fname.split('.')[:-1])
        df = pd.read_fwf(pth, widths=widths, nrows=n, names=dd.id.values)
    elif fname.endswith('.gz'):
        df = pd.read_fwf(fname, widths=widths, nrows=n, names=dd.id.values,
                         compression='gzip')
    else:
        raise IOError('Was the thing even zipped?')
    return df, dd


def dedup_cols(df):
    """
    Will append a suffix to the index keys which are duplicated.

    I'm hitting multiple PADDING's.
    """
    idx = df.columns
    dupes = idx.get_duplicates()
    print("Duplicates: {}".format(dupes))

    return df.T.drop(dupes).T


def pre_process(df):
    df = dedup_cols(df)
    # May want to be careful with index here.
    # forcing numeric chops of leading zeros.
    df = df.convert_objects(convert_numeric=True)
    df = df.set_index(['HRHHID', 'HUHHNUM', 'PULINENO'])
    return df


class file_handler(object):
    """
    Just useful the first time since I'm rezipping as gzip, which can be parsed
    on the fly.

    Sorry windows; Replace subprocess.call with appropriate utility.
    Implements context manager that decompresses and cleans up once
    the df has been read in.

    Example:
        fname = 'Volumes/HDD/Users/tom/DataStorage/CPS/monthly/cpsb0201.Z'
        with file_handler(fname):
            pre_process(df)


    """
    def __init__(self, fname, force=False):
        if os.path.exists(fname):
            self.fname = fname
            self.force = force
        else:
            raise IOError("The File does not exist.")

    def __enter__(self):
        if self.fname.endswith('.Z'):
            subprocess.call(["uncompress", "-v", self.fname])
        elif self.fname.endswith('.gzip'):
            if self.force:
                subprocess.call(["gzip", "-d", self.fname])
            else:
                print('Skipping decompression.')
        elif self.fname.endswith('.zip'):
            subprocess.call(["7z", "x", self.fname])
        self.compname = self.fname.rstrip('.Z')

    def __exit__(self, exc_type, exc_val, exc_tb):
        subprocess.call(["gzip", self.compname])
        if self.fname.endswith('.gz') and self.force:
            os.remove(self.fname.replace('.gz', '.txt'))
