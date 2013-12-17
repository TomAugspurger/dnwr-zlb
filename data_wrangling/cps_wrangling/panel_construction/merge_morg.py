import os

import pandas as pd
import numpy as np

from make_hdf_store import standardize_ids


def main():
    idx_cols = ['qmonth', 'HRHHID', 'HRHHID2', 'PULINENO']

    with pd.get_store('/Volumes/HDD/Users/tom/DataStorage/CPS/analyzed/clean.h5') as store:
        trim = store.select('trim')

    trim = trim.reset_index()
    trim['HRHHID'] = trim['HRHHID'].astype(int)
    trim['PULINENO'] = trim['PULINENO'].astype(int)
    trim = trim.set_index(idx_cols)

    base_path = "/Volumes/HDD/Users/tom/DataStorage/CPS/morg"
    dtas = [x for x in os.listdir(base_path) if int(x.split('.')[0]) >= 1996]

    for filename in dtas:
        year = filename.split('.')[0]

        if year == '2010':
            # malformed dta file
            # just do it in R
            df = pd.read_csv('/Volumes/HDD/Users/tom/DataStorage/CPS/morg/2010.csv')
        else:
            df = pd.read_stata(os.path.join(base_path, filename))

        quarter = df.intmonth.replace({1: '01', 2: '01', 3: '01',
                                       4: '04', 5: '04', 6: '04',
                                       7: '07', 8: '07', 9: '07',
                                       10: '10', 11: '10', 12: '10'})
        q = pd.to_datetime(df.year.astype('str') + quarter + '01', format='%Y%m%d')
        df.loc[:, 'qmonth'] = q
        df = df.rename(columns={'hhid': 'HRHHID', 'hrhhid2': 'HRHHID2',
                                'lineno': 'PULINENO', 'serial': 'HRSERSUF',
                                'hrsample': 'HRSAMPLE', 'hhnum': 'HUHHNUM'})
        if int(year) == 2004:
            df = df.dropna(subset=['HRHHID2'])
        if int(year) <= 2004:
            df.loc[:, 'HRHHID2'] = standardize_ids(df)

        if not df.HRHHID.dtype == np.int64:
            df.loc[:, 'HRHHID'] = df.HRHHID.str.lstrip('0').astype(int)
        df = df.dropna(subset=['HRHHID2'])
        df['HRHHID2'] = df['HRHHID2'].astype(int)

        df = df.set_index(idx_cols)
        df = df[(df['age'] > 18) & (df['age'] < 65)]

        merged = df.join(trim, how='left', rsuffix='_r')

        common_cols = ['Q1', 'Q2', 'Q3', 'age', 'earnwt', 'earnwke', 'earnings', 'earnhre',
                       'edu', 'ethnic', 'expr', 'flow', 'industry',
                       'labor_status', 'lfsr94', 'marital', 'married_d', 'month',
                       'nonemployed_history', 'unemployed_history', 'either_history',
                       'og_weight', 'paidhre', 'productivity', 'quarter', 'race', 'race_d',
                       'same_employer', 'sex', 'sex_d', 'uhourse', 'hourslw',
                       'edu_bin']

        if int(year) <= 1997:
            extra_cols = []
        elif 1998 <= int(year) <= 1999:
            extra_cols = ['ihigrdc']
        else:
            extra_cols = [u'ihigrdc', u'ind02']
        merged = merged[common_cols + extra_cols]
        hours = merged[['uhourse']].copy()

        hours.update(merged[['hourslw']].rename(columns={'hourslw': 'uhourse'}),
                     overwrite=False)
        merged.loc[:, 'uhourse'] = hours
        assert merged.index.is_unique

        merged = merged.dropna(subset=['flow'])

        with pd.get_store('/Volumes/HDD/Users/tom/DataStorage/CPS/analyzed/clean.h5') as store:
            merged.to_hdf(store, 'y' + year, format='t', append=False)

        print((year, len(merged.labor_status)))


def merge():
    dfs = []
    with pd.get_store('/Volumes/HDD/Users/tom/DataStorage/CPS/analyzed/clean.h5') as store:

        for k, v in store.iteritems():
            if k.startswith('/y'):
                df = store.select(k)

                df = df[(df['age'] > 18) & (df['age'] < 65)]

                dfs.append(df)

    df = pd.concat(dfs)
    assert df.index.is_unique
    with pd.get_store('/Volumes/HDD/Users/tom/DataStorage/CPS/analyzed/clean.h5') as store:
        df.to_hdf(store, 'merged', format='t', append=False)

if __name__ == '__main__':
    main()
