from datetime import datetime

import arrow
import numpy as np
import pandas as pd


def sane_names(df):
    names = {'PRTAGE': 'age', 'PTDTRACE': 'race', 'PESEX': 'sex',
             'PEMARITL': 'married', "PRERNWA": "earnings", "HRYEAR4": "year",
             'HRMONTH': "month", "PEMLR": 'labor_status', 'PEEDUCA': 'edu',
             'PEHRUSL1': 'hours'}
    if isinstance(df, pd.Panel):
        return df.rename(items=names, minor_axis=names)
    return df.rename(columns=names)


def convert_names(code, way='cps'):
    """
    Give a key. `way`='cps' goes from my names to CPS codes. `way`='sane'
    is the inverse.
    """
    d = {'age': 'PRTAGE',
         'earnings': 'PRERNWA',
         'edu': 'PEEDUCA',
         'hours': 'PEHRUSL1',
         'labor_status': 'PEMLR',
         'married': 'PEMARITL',
         'month': 'HRMONTH',
         'race': 'PTDTRACE',
         'sex': 'PESEX',
         'year': 'HRYEAR4'}

    if way == 'sane':
        d = {v: k for k, v in d.iteritems()}

    return d[code]

def get_useful(df):
    df = sane_names(df)
    # TODO: grab cols from convert_names.keys()
    # or both from a common function.
    cols = ['age', 'race', 'sex', 'married', 'earnings', 'year', 'month',
            'labor_status', 'edu', 'hours', 'timestamp']
    if isinstance(df, pd.Panel):
        res = df.loc[cols]
    else:
        res = df[cols]
    return res

def replace_categorical(df, inverse=False):
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
    if inverse:
        replacer = {v: k for k, v in replacer.iteritems()}

    # df.replace(replacer)  # should work. bug
    for k, v in replacer.iteritems():
        df[k] = df[k].replace(v)
    return df


def add_flows(s1, s2, categorical=True):
    """
    take a 2 column dataframe.

    returns a series or str/nan indexed like the 2.
    """
    # 1 & 2 is employed
    # 3 & 4 is unemployed
    # 1 - 4 is labor force
    if categorical:
        c1 = ['employed', 'absent']
        c2 = ['layoff', 'looking']
        c3 = ['retired', 'disabled', 'other']
    else:
        c1 = [1, 2]
        c2 = [3, 4]
        c3 = [4, 5, 6]

    ee = s1.isin(c1) & s2.isin(c1)
    eu = s1.isin(c1) & s2.isin(c2)
    en = s1.isin(c1) & s2.isin(c3)
    ue = s1.isin(c2) & s2.isin(c1)
    uu = s1.isin(c2) & s2.isin(c2)
    un = s1.isin(c2) & s2.isin(c3)
    ne = s1.isin(c3) & s2.isin(c1)
    nu = s1.isin(c3) & s2.isin(c2)
    nn = s1.isin(c3) & s2.isin(c3)

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
    labor = wp['labor_status']
    d = {1: pd.Series(np.nan, labor[1].index)}
    for i in range(1, 8):
        s1 = labor[i]
        s2 = labor[i+1]
        d[i+1] = add_flows(s1, s2, categorical=False)
    d = pd.DataFrame(d)
    if inplace:
        wp['flow'] = d
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


def transform_date(month, to='earn'):
    """
    give month and transform to either:
        - the resulting earning (if to=`earn`)
        - the generating panel (if to=`panel`)
    """
    month = month.strip('/').strip('m')
    ar = arrow.get(month, 'YYYY_MM')
    if to == 'earn':
        out = ar.replace(months=15)
    elif to == 'panel':
        out = ar.replace(months=-15)
    else:
        raise ValueError
    return out.strftime('%Y_%m')


def panel_to_frame(wp):
    df = wp.transpose(1, 0, 2).to_frame(filter_observations=False).T.stack().reset_index().set_index(
        ['minor', 'HRHHID', 'HUHHNUM', 'PULINENO']).sort_index()
    return df


def get_from_earn(month, earn_store, stat='median'):
    df = earn_store.select('m' + month)
    df = get_useful(df)
    df = df.replace(-1, np.nan)
    df = replace_categorical(df)
    x = df['earnings'].unstack()
    return getattr((x[8] - x[4]), 'median')()


def quantile_list(df, quantiles):
    # this will work for both Series and DataFrames
    return type(df)({q: df.quantile(q) for q in quantiles})
