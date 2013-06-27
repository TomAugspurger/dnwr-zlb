"""
We have data dictionaries parsed in an HDFStore.
We have the cps zipped repo.

Combine for an HDFStore of CPS tables.

Note on layout:

cps_store/
    monthly/
        dd/
        data/
            mYYYY_MM
            mYYYY_MM

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
import re
import os
import json
import subprocess

import pathlib
import pandas as pd

#-----------------------------------------------------------------------------
# Helper Functions


def writer(df, name, repo_path):
    """
    Write the dataframe to the HDFStore. Non-pure.

    Parameters
    ----------
    df: DataFrame to be writter
    name: name in the table; will be prepended with '/monhtly/data/m'
    repo_path: path to the store.  Get from settings.

    Returns
    -------
    None - IO
    """
    with pd.get_store(repo_path) as store:
        try:
            store.remove('/monthly/data/' + name)
        except KeyError:
            pass
        store.append('/monthly/data/' + name)


def dedup_cols(df):
    """
    Drops columns that are duplicated.  Have to transpose for unknown reason.
    I'm hitting multiple PADDING's, that should be it.
    
    Parameters
    ----------
    df : DataFrame

    Returns
    df : Same DataFrame, less the dupes.
    """
    idx = df.columns
    dupes = idx.get_duplicates()
    print("Duplicates: {}".format(dupes))

    return df.T.drop(dupes).T


def pre_process(df, ids):
    """
    Get DataFrame ready for writing to store.

    Makes (hopefully) unique index and makes types numeric.

    Parameters
    ----------
    df  : DataFrame
    ids : Columns to be used as the index.

    Returns
    -------
    df : DataFrame
    """
    df = dedup_cols(df)
    # May want to be careful with index here.
    # forcing numeric chops of leading zeros.
    df = df.convert_objects(convert_numeric=True)
    df = df.set_index(ids)
    return df


class FileHandler(object):
    """
    Takes care of file system details when working on the zipped files.
    Handles .Z, .zip, .gzip.
    

    Sorry windows; Replace subprocess.call with appropriate utility.
    Implements context manager that decompresses and cleans up once
    the df has been read in.

    Parameters
    ----------

    fname : String, path to zipped file.
    force : Bool, default False.  If True will unzip and rezip the gzip file.

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
            dir_name = '/'.join(self.fname.split('/')[:-1])
            # Unzipping gives new name; can't control.  Get diff
            current = {x for x in pathlib.Path(dir_name)}
            subprocess.call(["unzip", self.fname, "-d", dir_name])
            new = ({x for x in pathlib.Path(dir_name)} - current).pop()
            self.new_path = str(new)
        self.compname = self.fname.split('.')[0]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        subprocess.call(["gzip", self.compname])  # Gives warnings on zips.
        if self.fname.endswith('.gz') and self.force:
            os.remove(self.fname.replace('.gz', '.txt'))
        if self.fname.endswith('.zip'):
            os.remove(self.new_path)


def get_dd(fname, settings=None):
    """
    Helper to get the data dictionary associated with a given month's filename.

    Parameters
    ----------
    fname: str, either full or shortened should work.
    settings: str or None.  Path to the settings file

    Returns
    -------
    dd: str, name of data dictionary. 

    """
    if settings is None:
        settings = json.load(open('info.txt'))
    just_name = fname.split('/')[-1].split('.')[0]
    return settings['month_to_dd_by_filename'][just_name]


def get_id(target, store):
    """
    Target is a str, e.g. HHID; This finds all this little
    idiosyncracies.
    """
    for key in store.keys():
        dd = store.select(key)
        yield key, dd.id[dd.id.str.contains(target)]


def find_attr(attr, fields):
    """
    Dictionary may lie.  Check here.

    Parameters
    ----------
    attr: str; e.g. "AGE", "RACE", "SEX"
    df: Index; probably columns of the dataframe.

    Returns
    -------

    List of strs possible matches.
    """

    match_with = re.compile(r'\w*' + attr + r'\w*')
    maybe_matches = (match_with.match(x) for x in fields)
    return [x.string for x in filter(None, maybe_matches)]


def drop_invalid_indicies(df, dd_name=None):
    """
    df: Full dataframe
    dd_name: name to lookup
    """
    # Generalize to getting the -1 (invlaid) from settings.
    valids = df[~(df.apply(lambda x: x.name[2] == -1, axis=1))]
    return valids


def drop_duplicates_index(df, dupes=None):
    """Isn't a method on the dataframe oddly"""
    if dupes is None:
        dupes = df.index.get_duplicates()
    return df.ix[~(df.index.isin(dupes))]

#-----------------------------------------------------------------------------
if __name__ == '__main__':
    import sys

    try:
        settings = json.load(open(sys.argv[1]))
    except IndexError:
        settings = json.load(open('info.txt'))

    raw_path  = pathlib.Path(str(settings['raw_monthly_path']))
    base_path = settings['base_path']
    repo_path = settings['repo_path']
    dds       = pd.HDFStore(settings['store_path'])

    for month in raw_path:
        try:
            s_month = str(month)
            name = s_month.split('.')[0]
            just_name = month.parts[-1].split('.')[0]
            dd_name = settings["month_to_dd_by_filename"][just_name]
            ids = settings["dd_to_ids"][dd_name]
            dd = dds.select('/monthly/dd/' + dd_name)
            widths = dd.length.tolist()
        except KeyError:
            print(month)
            continue

        if s_month.endswith('.gz'):
            df = pd.read_fwf(name + '.gz', widths=widths,
                             names=dd.id.values, compression='gzip', nrows=10)
        else:
            with FileHandler(s_month) as handler:
                try:
                    name = handler.new_path
                except AttributeError:
                    pass
                df = pd.read_fwf(name, widths=widths, names=dd.id.values, nrows=None)
        df = pre_process(df, ids=ids).sort_index()
        cols = settings['dd_to_vars'][dd_name].values()
        if df.index.is_unique:
            out_name = settings['file_to_iso8601'][just_name]
            writer(df, name=out_name, repo_path=repo_path)

        else:
            dupes = df.index.get_duplicates()
            gen = (df[cols].xs(x) for x in dupes)

            with open('non_unique.txt', 'a') as f:
                f.write(s_month)

