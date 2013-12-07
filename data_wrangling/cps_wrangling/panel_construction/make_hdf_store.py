"""
We have data dictionaries parsed in an HDFStore.
We have the cps zipped repo.

All of these programs (make_*) should accept an optinal param
(just sys.argv for now, argparser later) that will restrict
the months to a subset. Read from a file for now. the format must be
YYYY_MM

Storing the following columns for each month:


common = {"PRTAGE", "HRMIS", "HRYEAR4", "PESEX", "HRMONTH", "PTDTRACE",
          "PEMLR", "PRERNWA", "PTWK", "PEMARITL", "PRDISC",
          "HEFAMINC", "PTDTRACE", "HWHHWGT", "PEERNHRY", "HRMIS"}




Here's a dict of CPS names to my names.

names = {'PRTAGE': 'age', 'PTDTRACE': 'race', 'PESEX': 'sex',
         'PEMARITL': 'married', "PRERNWA": "earnings", "HRYEAR4": "year",
         'HRMONTH': "month", "PEMLR": 'labor_status'}

The philosophy here is to raise early and raise often. If columns aren't
aligning then we should raise immediatly.

We'll also be strict about what makes it in to the panel. If *ANY* identifier
is missing then that line gets dropped. This can be revisted in the future.

Convention:

* all months will be refered to by YYYY_MM. The only excpetion is
in the keys to stores, which will be 'mYYYY_MM'
"""
from datetime import datetime
from difflib import get_close_matches
import itertools
import json
import os
import re

import arrow
import pathlib
import pandas as pd
from matplotlib.cbook import flatten
import numpy as np

#-----------------------------------------------------------------------------
# File Handling / IO
#-----------------------------------------------------------------------------


def writer(df, name, store_path, settings, start_time, overwrite=True):
    """
    Write the dataframe to the HDFStore. Non-pure.

    Parameters
    ----------
    df: DataFrame to be writter
    name: name in the table; will be prepended with '/monhtly/data/m'
    store_path: path to the store.  Get from settings.

    Returns
    -------
    None - IO
    """
    with pd.get_store(store_path) as store:
        store.append(name, df, append=False)

    logname = name.lstrip('m')
    with open(settings["make_hdf_store_completed"], 'a') as f:
        f.write('{},{},{}\n'.format(logname, start_time, arrow.utcnow()))


#-----------------------------------------------------------------------------
# Duplicate Handling
#-----------------------------------------------------------------------------


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
    if not df.columns.is_unique:
        dupes = df.columns.get_duplicates()
        print("Duplicates: {}".format(dupes))
        return df.drop(dupes, axis=1)
    else:
        return df


def pre_process(df, ids, std_ids=False):
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

    if std_ids:
        df['HRHHID2'] = standardize_ids(df)
        df = df.drop('HRSAMPLE', axis=1)
        df = df.drop('HRSERSUF', axis=1)

    df = df.convert_objects(convert_numeric=True)
    try:
        df[df == -1] = np.nan
    except TypeError:
        df = df.replace(-1, np.nan)
    df = df.replace('-', np.nan)

    df = df.loc[~(pd.isnull(df[ids]).any(1)), :]
    df = df.set_index(ids)

    if "FILLER" in df.columns:
        df = df.drop("FILLER", axis=1)
    if "PADDING" in df.columns:
        df = df.drop("PADDING", axis=1)
    return df


def post_process(df):
    """
    Stuff that depends on standardize_cols
    """
    assert len(df.HRYEAR4.dropna().unique()) == 1
    assert len(df.HRMONTH.dropna().unique()) == 1
    year = int(df.HRYEAR4.iloc[0])
    month = int(df.HRMONTH.iloc[0])

    df['timestamp'] = datetime(year, month, 1)
    df['PEHRUSL1'] = df['PEHRUSL1'].replace(-4, np.nan)  # Hours vary
    return df


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


def make_regex(style=None):
    if style is None:
        return re.compile(r'(\w{1,2}[\$\-%]\w*|PADDING)\s*CHARACTER\*(\d{3})\s*\.{0,1}\s*\((\d*):(\d*)\).*')
    elif style is 'aug2005':
        return re.compile(r'(\w+)[\s\t]*(\d{1,2})[\s\t]*(.*?)[\s\t]*\(*(\d+)\s*-\s*(\d+)\)*$')
    elif style is 'jan1998':
        return re.compile(r'D (\w+) \s* (\d{1,2}) \s* (\d*)')


def handle_dupes(df, settings):
    """
    Get subset of df that doesn't have duplicated index.

    Parameters
    ----------

    df : DataFrame.  Index should have name
    settings: dict.

    Returns
    -------

    deduped: DataFrame with duplicate indicies removed.
    """
    dupes = df.index.get_duplicates()
    parts = (df.xs(x) for x in dupes)
    deduped = drop_duplicates_index(df, dupes=dupes)

    dupe_file = settings['dupe_path'] + df.index.name + '.csv'
    dir_ = '/' + os.path.join(*dupe_file.split('/')[:-1])
    if not os.path.exists(dir_):
        os.mkdir(dir_)
    with open(dupe_file, 'w') as f:
        header = ','.join(map(str, df.index.names) + df.columns.tolist()) + '\n'
        f.write(header)

    for part in parts:
        part.to_csv(dupe_file, mode='a', header=False)

    print("Deduplicated {}".format(df.index.name))
    return deduped


def handle_89_pt2(df):
    """After numeric.  Get only adult records."""
    df = df[df['HhRECTYP'] == 1]
    return df

#-----------------------------------------------------------------------------
# Helpers
#-----------------------------------------------------------------------------


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
        settings = json.load(open('settings.txt'))
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


def get_definition(code, dd_path=None, style=None):
    """
    Get the definition for a code.

    Maybe add option to pass dd_name with a lookup from settings as convinence.
    """

    regex = make_regex(style)

    def dropper(line):
        maybe_match = regex.match(line)
        try:
            line_code = maybe_match.groups()[0]
            return line_code
        except AttributeError:
            return None

    def get_def(dd):
        """
        Next of dd is the line you start with.  Now consume up to next match.
        """
        top_line = [''.join(list(next(dd)))]
        rest = it.takewhile(lambda x: regex.match(x) is None, dd)
        rest = [''.join(x) for x in rest]
        top_line.append(rest)
        return top_line

    with open(dd_path) as dd:
        gen = it.dropwhile(lambda x: dropper(x) != code, dd)
        definition = get_def(gen)
        return definition


def grouper(dict_, ):
    """
    Another helper for checking the fields.

    Parameters
    ----------

    dict_: dictionary

    Returns
    -------
    matched : list
    remainder : list
    """
    # Convert dict to sorted list of items
    list_ = sorted(dict_.items(), key=lambda x: x[1])

    # Group by value of tuple
    groups = itertools.groupby(list_, key=lambda x: x[1])

    # Pull out matching groups of items, and combine items
    # with no matches back into a single dictionary

    dict_groups = {key: list(group) for key, group in groups}
    try:
        trues = [x[0] for x in dict_groups[True]]
    except KeyError:
        trues = None
    try:
        falses = [x[0] for x in dict_groups[False]]
    except KeyError:
        falses = None
    return {True: trues, False: falses}


def check_fieldname(field, settings, dd=None, store_path=None):
    """
    Helper to see if a given field is in a data dictionary.

    Parameters
    ----------

    field : str or list.  Column in df or row in dd.id.
    settings : JSON settings file.
    dd : DataFrame.  With col containing fields.
    store_path : path within store to dd. Only used if dd is None.

    Returns
    -------

    grouped : dict of True : [], False: []

    Example
    -------

    >>> res = check_fieldname(flatten(
        settings['dd_to_vars']['may2012'].values()), settings, dd=dd)

    >>> {x: get_close_matches(x, dd.id) for x in res[False]}
    """
    if dd is None:
        with pd.get_store(settings['store_path']) as store:
            dd = store.select(store_path)

    if isinstance(field, str):
        fields = [field]
    else:
        fields = list(field)

    ungrouped = {x: x in dd.id.values for x in fields}
    try:
        grouped = grouper(ungrouped)
        return grouped
    except KeyError as e:
        print(e)
        return None


def find_attr(attr, fields=None, dd=None, settings=None):
    """
    Dictionary may lie.  Check here.

    Parameters
    ----------
    attr: str; e.g. "AGE", "RACE", "SEX"
    fields: array-like; probably columns of the dataframe.
    dd : str or DataFrame; path inside the store or the DD itself.
    settings: dict with "store_path"
    Returns
    -------

    List of strs possible matches.
    """
    if settings is None:
        settings = json.load(open('settings.txt'))

    with pd.get_store(settings["store_path"]) as store:
        if fields is not None and dd is not None:
            raise ValueError('One of fields and dd must be specified.')
        elif fields is None and dd is None:
            raise ValueError('One of fields and dd must be specified.')
        elif dd and isinstance(dd, str):
            dd = store.select(dd)
            fields = dd.id.tolist()
        elif dd and isinstance(dd, pd.DataFrame):
            fields = dd.id.tolist()

    match_with = re.compile(r'[\w|$%\-]*' + attr + r'[\w|$%\-]*')
    maybe_matches = (match_with.match(x) for x in fields)
    return [x.string for x in filter(None, maybe_matches)]


def run_one(path, settings, n=10):
    """
    Helper to get a single month's data and data dictonary.

    Parameters
    ----------

    path : str. path to the raw data.
    settings : dict. From the settings JSON file.
    n : int.  Number of rows to read.  None if you want it all.

    Returns
    -------

    df, dd : tuple with DataFrame and Data Dictonary.
    """
    month = pathlib.Path(path)
    just_name, out_name, s_month, name, dd_name = name_handling(month, settings)

    with pd.get_store(settings['store_path']) as dds:
        dd = dds.select('/monthly/dd/' + dd_name)

    ids = settings["dd_to_ids"][dd_name]
    widths = dd.length.tolist()

    if s_month.endswith('.gz'):
        df = pd.read_fwf(name + '.gz', widths=widths,
                         names=dd.id.values, compression='gzip', nrows=n)
    else:
        with FileHandler(s_month) as handler:
            try:
                name = handler.new_path
            except AttributeError:
                pass
            df = pd.read_fwf(name, widths=widths, names=dd.id.values, nrows=n)

    df = pre_process(df, ids=ids).sort_index()

    if dd_name in ['jan1989', 'jan1992']:
        df = handle_89_pt2(df)

    # subset = get_subset(df, settings=settings)
    df.index.name = out_name

    return df, dd
#-----------------------------------------------------------------------------
# Logging
#-----------------------------------------------------------------------------


def log_and_store(df):
    with open('ready_to_store.txt', 'w') as f:
        if df.index.is_unique:
            f.write('Ready: {}\n'.format(df.index.name))
        else:
            dupes = df.index.get_duplicates()
            f.write('Dupes: {}\n'.format(df.index.name))
            f.write("\t\t {}".format(dupes))


def get_skips(file_):
    try:
        with open(file_, 'r') as f:
            skips = [line.split(',')[0].rstrip() for line in f]
    except IOError:
        with open(file_, 'w') as f:
            skips = []

    return skips


#-----------------------------------------------------------------------------
# MISC
#-----------------------------------------------------------------------------
def get_subset(df, settings, dd_name, quiet=False):
    """
    Select only those columns specified under settings.
    Optionaly

    Parameters
    ----------

    df : DataFrame
    settings : dictionary with "dd_to_vars" column
    dd_name : str. for the lookup.
    quiet: Bool.  If True will print, but not raise, on some columns
    from settings not being in the df's columns.

    Returns
    -------

    subset : DataFrame.
    """
    cols = {x for x in flatten(settings["dd_to_vars"][dd_name].values())}
    good_cols = {x for x in flatten(settings["dd_to_vars"]["jan2013"].values())}
    all_cols = cols.union(good_cols)
    subset = df.columns.intersection(all_cols)

    if not quiet:
        print("Implicitly dropping {}".format(cols.symmetric_difference(subset)))

    return df[subset]


def name_handling(month, settings, skip=True):
    """
    Handle all the name stuff for a function run.

    Parameters
    ----------

    month: pathlib.Path object.  Path to the data.
    settings : JSON dictionary.

    Returns
    -------

    just_name : str; filename without any leading directories.
    out_name : str; For use in store.  'm' + iso8601.
    s_month : str; same as month but in str form.
    name : str; month without the file extension.
    dd_name : str; data dictionary name.
    """
    just_name = month.parts[-1].split('.')[0]
    if just_name == '' or month.is_dir():
        return just_name, _, _, _, _

    out_name = arrow.get(just_name, 'YYYY_MM').strftime('m%Y_%m')
    s_month = str(month)
    name = s_month.split('.')[0]
    dd_name = settings["month_to_dd"][''.join(just_name.split('_'))]

    return just_name, out_name, s_month, name, dd_name


def standardize_cols(df, dd_name, settings):
    """
    Rename cols in df according to the spec in settings for that year.

    standaradize_cols :: df -> str -> dict -> df
    """
    renamer = settings["col_rename_by_dd"][dd_name]
    df = df.rename(columns=renamer)

    common = {"PRTAGE", "HRMIS", "HRYEAR4", "PESEX", "HRMONTH", "PTDTRACE",
              "PEMLR", "PRERNWA", "PTWK", "PEMARITL", "PRDISC",
              "HEFAMINC", "PTDTRACE", "HWHHWGT", "PEERNHRY", "HRMIS"}
    cols = set(df.columns.tolist())
    extra = cols - common
    missing = common - cols

    if missing:
        name = str(df.HRYEAR4.iloc[0]) + str(df.HRMONTH.iloc[0])
        key = ' '.join([str(arrow.utcnow()), name, 'missing'])
        d = {key: list(missing)}
        with open('make_hdf_store_log.json', 'a') as f:
            json.dump(d, f, indent=2)

    if extra:
        name = str(df.HRYEAR4.iloc[0]) + str(df.HRMONTH.iloc[0])
        key = ' '.join([str(arrow.utcnow()), name, 'extra'])
        d = {key: list(extra)}
        with open('make_hdf_store_log.json', 'a') as f:
            json.dump(d, f, indent=2)

    return df


def standardize_ids(df):
    """
    pre may2004 need to fill out the ids by creating HRHHID2 manually:
    (ignore the position values, this is from jan2013)

    HRHHID2        5         HOUSEHOLD IDENTIFIER (part 2) 71 - 75

         EDITED UNIVERSE:    ALL HHLD's IN SAMPLE

         Part 1 of this number is found in columns 1-15 of the record.
         Concatenate this item with Part 1 for matching forward in time.

         The component parts of this number are as follows:
         71-72     Numeric component of the sample number (HRSAMPLE)
         73-74     Serial suffix-converted to numerics (HRSERSUF)
         75        Household Number (HUHHNUM)

    NOTE: not documented by sersuf of -1 seems to map to '00'
    """
    import string

    hrsample = df['HRSAMPLE'].str.slice(1,)
    hrsersuf = df['HRSERSUF'].astype(str)
    huhhnum = df['HUHHNUM'].astype(str)

    sersuf_d = {a: str(ord(a.lower()) - 96).zfill(2) for a in hrsersuf.unique()
                if a in list(string.ascii_letters)}
    sersuf_d['-1.0'] = '00'
    sersuf_d['-1'] = '00'
    hrsersuf = hrsersuf.map(sersuf_d)  # 10x faster than replace
    hrhhid2 = hrsample + hrsersuf + huhhnum
    return hrhhid2.astype(np.int64)


def special_by_dd(keys):
    """All of these are inplace"""
    def expand_year(df, dd_name):
        """ For jan1989 - sep1995 they wrote the year as a SINGLE DIGIT"""
        if 'HRYEAR' in df.columns:
            k = 'HRYEAR'
        else:
            k = k = 'HdYEAR'
        last_digit = df[k].dropna().unique()[0]
        if last_digit >= 10:
            last_digit = last_digit % 10
        base_year = int(dd_name[-4:-1]) * 10
        df["HRYEAR4"] = base_year + last_digit
        df = df.drop(k, axis=1)
        return df

    def combine_age(df, dd_name):
        """For jan89 and jan92 they split the age over two fields."""
        df["PRTAGE"] = df["AdAGEDG1"] * 10 + df["AdAGEDG2"]
        df = df.drop(["AdAGEDG1", "AdAGEDG2"], axis=1)
        return df

    def align_lfsr(df, dd_name):
        """Jan1989 and Jan1999. LFSR (labor focrce status recode)
        had
           1 = WORKING
           2 = WITH JOB,NOT AT WORK
           3 = UNEMPLOYED, LOOKING FOR WORK
           4 = UNEMPLOYED, ON LAYOFF
           5 = NILF - WORKING W/O PAY < 15 HRS;
                      TEMP ABSENT FROM W/O PAY JOB
           6 = NILF - UNAVAILABLE
           7 = OTHER NILF
        newer ones have
           1   EMPLOYED-AT WORK
           2   EMPLOYED-ABSENT
           3   UNEMPLOYED-ON LAYOFF
           4   UNEMPLOYED-LOOKING
           5   NOT IN LABOR FORCE-RETIRED
           6   NOT IN LABOR FORCE-DISABLED
           7   NOT IN LABOR FORCE-OTHER
        this func does several things:
            1. Change 3 -> 4 and 4 -> 3 in the old ones.
            2. Change 5 and 6 to 7.
            2. Read retired from AhNLFREA == 4 and set to 5.
            3. Read ill/disabled from AhNLFREA == 2 and set to 6.
        Group 7 kind of loses meaning now.
        """
        # 1. realign 3 & 3
        status = df["AhLFSR"]
        # status = status.replace({3: 4, 4: 3})  # chcek on ordering

        status_ = status.copy()
        status_[status == 3] = 4
        status_[status == 4] = 3
        status = status_

        # 2. Add 5 and 6 to 7
        status = status.replace({5: 7, 6: 7})

        # 3. ill/disabled -> 6
        status[df['AhNLFREA'] == 2] = 6

        df['PEMLR'] = status
        df = df.drop(["AhLFSR", "AhNLFREA"], axis=1)
        return df

    def expand_hours(df, dd_name):
        """
        89 and 92 have a question for hours and bins. I goto midpoint of bin.

        Roughly corresponds to PEERNHRO

        A-EMPHRS    CHARACTER*002 .     (0357:0358)           LFSR=1 OR 2
           REASONS NOT AT WORK OR HOURS AT WORK
           -1 = NOT IN UNIVERSE
           WITH A JOB, BUT NOT AT WORK
           01 = ILLNESS
           02 = VACATION
           03 = BAD WEATHER
           04 = LABOR DISPUTE
           05 = ALL OTHER
           AT WORK
           06 = 1-4 HOURS
           07 = 5-14 HOURS
           08 = 15-21 HOURS
           09 = 22-29 HOURS
           10 = 30-34 HOURS
           11 = 35-39 HOURS
           12 = 40 HOURS
           13 = 41-47 HOURS
           14 = 48 HOURS
           15 = 49-59 HOURS
           16 = 60 HOURS OR MORE
        """
        hours = df['AhEMPHRS']
        hours_dic = {1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan, 5: np.nan,
                     6: 2, 7: 9.5, 8: 18, 9: 25.5, 10: 32, 11: 37, 13: 44,
                     15: 54}
        hours = hours.replace(hours_dic)
        df['PEERNHRO'] = hours
        df.drop("AhEMPHRS", axis=1)
        return df

    def combine_hours(df, dd_name):
        """
        For 89 and 92; "AdHRS1", "AdHRS2" combine to form "PEHRACTT"
        """
        fst = df['AdHRS1']
        snd = df['AdHRS2']
        df['PEHRACTT'] = fst * 10 + snd
        df = df.drop(["AdHRS1", "AdHRS2"], axis=1)
        return df

    func_dict = {"expand_year": expand_year, "combine_age": combine_age,
                 "expand_hours": expand_hours, "align_lfsr": align_lfsr,
                 "combine_hours": combine_hours}
    to_apply = filter(lambda x: x in keys, func_dict)
    filtered = {k: func_dict[k] for k in to_apply}
    return filtered


def append_to_store(month, settings, skips, dds, start_time):
    try:
        just_name, out_name, s_month, name, dd_name = (
            name_handling(month, settings))

        if just_name == '' or month.is_dir():  # . files or subdirs
            return None

        ids = settings["dd_to_ids"][dd_name]

        try:
            dd = dds.select(dd_name)
            widths = dd.length.tolist()
        except KeyError:
            print("No data dictionary for {}".format(out_name))
            with open(settings["store_log"], 'a') as f:
                f.write("FAILED {}. No data dictionary".format(out_name))
            return None

        df = pd.read_fwf(name + '.gz', widths=widths,
                         names=dd.id.values, compression='gzip')

        # starting in may04 the identifiers change.
        if dd_name in ["jan1994", "apr1994", "jun1995", "sep1995", "jan1998",
                       "jan2003"]:
            std_ids = True
        else:
            std_ids = False

        df = pre_process(df, ids=ids, std_ids=std_ids).sort_index()

        if dd_name in ['jan1989', 'jan1992']:
            df = handle_89_pt2(df)

        specials = special_by_dd(settings["special_by_dd"][dd_name])
        for func in specials:
            df = specials[func](df, dd_name)

        df = get_subset(df, settings=settings, dd_name=dd_name)
        df = standardize_cols(df, dd_name, settings)
        df = post_process(df)

        df.index.name = out_name

        #------------------------------------------------------------------
        # Ensure uniqueness
        if not df.index.is_unique:
            df = handle_dupes(df, settings=settings)
            assert df.index.is_unique
        #------------------------------------------------------------------
        # Writing
        store_path = settings['store_path']
        writer(df, name=out_name, store_path=store_path, settings=settings,
               start_time=start_time)
        print('Added {}'.format(out_name))
    except Exception as e:
        with open(settings["store_log"], 'a') as f:
            f.write('FAILED {} for reason {}.\n'.format(str(month), e))
        print('FAILED {} for reason {}.\n'.format(str(month), e))


def main():
    import sys

    try:
        some_months = sys.argv[1]
    except IndexError:
        print("Making for all months")
        some_months = False
    if some_months:
        with open('special_months_store.txt', 'r') as f:
            month_list = [line.strip() for line in f.readlines()]

    settings = json.load(open('settings.txt'))
    #-------------------------------------------------------------------------
    # setup
    start_time = arrow.utcnow()
    raw_path = pathlib.Path(str(settings['raw_monthly_path']))
    dds = pd.HDFStore(settings['dd_store_path'])

    skips = get_skips(settings['make_hdf_store_completed'])

    months = (month for month in raw_path if not month.parts[-1].startswith('.'))
    if some_months:
        months = (m for m in months if str(m).parts[-1].rstrip('gz') in month_list)

    months = (x for x in months if x.parts[-1].rstrip('.gz') not in skips)

    for month in months:
        append_to_store(month, settings, skips, dds, start_time=start_time)

    # cleanup
    dds.close()
    if some_months:
        with open('special_months_store.txt', 'w') as f:
            pass
#-----------------------------------------------------------------------------
if __name__ == '__main__':
    main()
