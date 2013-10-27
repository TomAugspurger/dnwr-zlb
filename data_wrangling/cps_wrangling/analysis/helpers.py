from datetime import datetime

import numpy as np
import pandas as pd


def sane_names(df, flattened=False):
    names = {'PRTAGE': 'age', 'PTDTRACE': 'race', 'PESEX': 'sex',
             'PEMARITL': 'married', "PRERNWA": "earnings", "HRYEAR4": "year",
             'HRMONTH': "month", "PEMLR": 'labor_status'}
    if flattened:
        names = add_suff(names)
    if isinstance(df, pd.Panel):
        return df.rename(minor_axis=names)
    return df.rename(columns=names)


def add_suff(xs):
    if isinstance(xs, dict):
        w1 = {k + '1': v + '1' for k, v in xs.iteritems()}
        w2 = {k + '2': v + '2' for k, v in xs.iteritems()}
        w1.update(w2)
        return w1
    else:
        w1 = map(lambda x: x + '1', xs)
        w2 = map(lambda x: x + '2', xs)
        return w1 + w2


def replace_categorical(df, flattened=False):
    sex = {1: "male", 2: "female"}
    race = {1: "White Only",
            2: "Black Only",
            3: "American Indian, Alaskan Native Only",
            4: "Asian Only",
            5: "Hawaiian/Pacific Islander Only",
            6: "White-Black",
            7: "White-AI",
            8: "White-Asian",
            9: "White-HP",
            10: "Black-AI",
            11: "Black-Asian",
            12: "Black-HP",
            13: "AI-Asian",
            14: "AI-HP",
            15: "Asian-HP",
            16: "W-B-AI",
            17: "W-B-A",
            18: "W-B-HP",
            19: "W-AI-A",
            20: "W-AI-HP",
            21: "W-A-HP",
            22: "B-AI-A",
            23: "W-B-AI-A",
            24: "W-AI-A-HP",
            25: "Other 3 Race Combinations",
            26: "Other 4 and 5 Race Combinations"}
    married = {1: "MARRIED, CIVILIAN SPOUSE PRESENT",
               2: "MARRIED, ARMED FORCES SPOUSE PRESENT",
               3: "MARRIED, SPOUSE ABSENT (EXC. SEPARATED)",
               4: "WIDOWED",
               5: "DIVORCED",
               6: "SEPARATED",
               7: "NEVER MARRIED"}
    # 1 & 2 is employed
    # 3 & 4 is unemployed
    # 1 - 4 is labor force
    labor_status = {1: "employed",
                    2: "absent",
                    3: "layoff",
                    4: "looking",
                    5: "retired",
                    6: "disabled",
                    7: "other"}

    replacer = {"sex": sex, "race": race, "married": married,
                "labor_status": labor_status}
    if flattened:
        d1 = {k + '1': v for k, v in replacer.iteritems()}
        d2 = {k + '2': v for k, v in replacer.iteritems()}
        replacer = {}
        replacer.update(d1)
        replacer.update(d2)
    return df.replace(replacer)


def add_flows(s1, s2):
    """
    take a 2 column dataframe.

    returns a series or str/nan indexed like the 2.
    """
    # 1 & 2 is employed
    # 3 & 4 is unemployed
    # 1 - 4 is labor force
    ee = s1.isin([1, 2]) & s2.isin([1, 2])
    eu = s1.isin([1, 2]) & s2.isin([3, 4])
    en = s1.isin([1, 2]) & s2.isin([5, 6, 7])
    ue = s1.isin([3, 4]) & s2.isin([1, 2])
    uu = s1.isin([3, 4]) & s2.isin([3, 4])
    un = s1.isin([3, 4]) & s2.isin([4, 5, 6])
    ne = s1.isin([5, 6, 7]) & s2.isin([1, 2])
    nu = s1.isin([5, 6, 7]) & s2.isin([3, 4])
    nn = s1.isin([5, 6, 7]) & s2.isin([5, 6, 7])

    flows = {"ee": ee, "eu": eu, "en": en, "ue": ue, "uu": uu, "un": un,
             "ne": ne, "nu": nu, "nn": nn}

    s = pd.Series(np.nan, index=s1.index)
    for k, v in flows.iteritems():
        s.loc[v] = k
    return s


def add_flows_panel(wp, inplace=True):
    """wrapper for add_flows.
    The flow in 2 is the flow *from* 1 *to* 2. 1 will contain NaNs.

    If not doing it inplace, assings with wp.loc[:, :, 'flow'] = returned
    """
    labor = wp.minor_xs('labor_status')
    d = {1: pd.Series(np.nan, labor[1].index)}
    for i in range(1, 8):
        s1 = labor[i]
        s2 = labor[i+1]
        d[i+1] = add_flows(s1, s2)
    d = pd.DataFrame(d)
    if inplace:
        wp.loc[:, :, 'flow'] = d
        return wp
    else:
        return wp


def clean_no_lineno(df):
    """drop if lineno is -1"""
    # TODO: why not just df.dropna(how='all', axis=0)?
    idx_names = df.index.names
    x = df.reset_index()
    # good_idx_iloc = x[~pd.isnull(x[idx_names]).any(1)].index
    df = (x.loc[~(x['PULINENO'] == -1)]).set_index(idx_names)
    return df


def prep_df(df, drop=False):
    """Collapse HRYEAR4 and HRMONTH into timestamp"""
    df = df.dropna(how='all', axis=0)
    df = df.convert_objects(convert_numeric=True)
    df['year'] = df.year.astype(np.int64)
    df['month'] = df.month.astype(np.int64)
    try:
        df['timestamp'] = df.apply(lambda row: datetime(row['year'], row['month'], 1),
                                   axis=1)
    except TypeError:
        df['timestamp'] = df.apply(lambda row: datetime(int(row['year']),
                                   int(row['month']), 1), axis=1)
    return df


def labor_status_value_counts(cps_store, month):
    """
    Value counts of labor status straight from cps_store. Give month
    name without leading m ('2010_01').  Sets name of the resulting series
    to month. Concat together with pd.concat([m1, ... m2], axis=1)
    """
    cols = ['HRYEAR4', 'PTDTRACE', 'PEMARITL', 'PRTAGE', 'PRERNWA', 'PEMLR',
            'PESEX', 'HRMONTH']
    useful = ['age', 'year', 'month', 'labor_status', 'earnings', 'race', 'sex', 'married']
    df = sane_names(cps_store.select('/monthly/data/m' + month, columns=cols))[useful]
    df = clean_no_lineno(df)
    df = df.dropna(how='all', axis=[0, 1])
    clean = df.dropna(how='all', axis=0)['labor_status']
    labels = {1: 'employed', 2: 'employed', 3: 'unemployed', 4: 'unemployed', 5: 'nilf', 6: 'nilf', 7: 'nilf'}
    cts = clean.replace(labels).value_counts().T
    # cts = cts.sort_index().T
    date = datetime(int(df.year.iloc[0]), int(df.month.iloc[0]), 1)
    cts.name = date
    return cts
