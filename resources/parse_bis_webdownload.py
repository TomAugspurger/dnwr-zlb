import itertools as it

import pandas as pd

from date_parsers import year_quarter_bis
#-----------------------------------------------------------------------------
# Helpers


def splitter(x, labels):
    parts = x.split(':')
    labels[parts[0]] = parts[1]
    return parts[0]


def get_group(df, ctry):
    gr = df.ix[ctry].unstack('counter').xs(('B', 'S'),
                                           level=('bank_type', 'claim_type'))
    return gr


def get_top(gr, n=10):
    # may need to gr.stack().unstack('quarter')
    return gr.sort(columns=[gr.columns[0]], ascending=False).index[:n]

#-----------------------------------------------------------------------------
# IO and Formatting
path = 'WEBSTATS_CIBL_IB_DATAFLOW-1368108852812.csv'

df = pd.read_csv(path, skiprows=12, parse_dates=['Quarter'],
                 date_parser=year_quarter_bis)


df[df.columns[:-1]] = df[df.columns[:-1]].fillna(method='ffill')

rename_cols = {'Reporting country': 'reporter',
               'Counterparty location': 'counter',
               'Type of reporting banks': 'bank_type',
               'Type of claim, counterparty sector and other breakdowns':
                                        'claim_type',
               'Quarter': 'quarter',
               'Unnamed: 5': 'value'}
df = df.rename(columns=rename_cols)

labels = {}
for col in df.columns[:4]:
    df[col] = df[col].apply(splitter, args=([labels]))

df = df.set_index(['reporter', 'counter', 'bank_type', 'claim_type',
                   'quarter'])
#-----------------------------------------------------------------------------

interest = ['GR', 'ES', 'IE', 'PT', 'CY', 'EE']
lenders = ['AT', 'BE', 'CH', 'CN', 'CY', 'CZ', 'DE', 'DK', 'ES', 'FI', 'FR',
           'GB', 'GR', 'IE', 'IT', 'JP', 'LU', 'NL', 'NO', 'LI', 'PL', 'PT',
           'RU', 'SE', 'SI', 'SK', 'US']
fewer = ['AT', 'CH', 'CN', 'DE', 'DK', 'ES', 'FR',
         'GB', 'GR', 'IE', 'IT', 'JP', 'LU', 'NL', 'PL', 'PT',
         'RU', 'US', 'CY']

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# Playing around: A matrix repr.
sub = df.xs(('B', 'S', '2012-10-01'),
            level=('bank_type', 'claim_type', 'quarter'))
s = sub.unstack()
value_cycle = it.cycle(['value'])
mat = s[zip(value_cycle, s.index)]
