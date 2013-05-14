# Japaneese Data
# From FRED
import numpy as np
import pandas as pd
from pandas.io.data import DataReader
import matplotlib.pyplot as plt

from parse_imf import parse_imf

series = ['JPNCPIALLMINMEI',  # CPI for all items
          'JPNRGDPQDSNAQ',    # nominal GDP
          'JPNURNAA',         # Adjusted Unemployment Rate
          'MYAGM2JPM189N',    # M2
          'INTGSTJPM193N'     # T-bill rate
          ]


def float_(x):
    try:
        return float(x)
    except:
        return np.nan

cpi, gdp, urate, m2, tbill = [DataReader(x, data_source='fred', start='1-1-1940') for x in series]
tbill = tbill.applymap(float_)

jf = pd.concat([cpi, gdp, urate, m2, tbill], axis=1)
jf = jf.rename(columns={'JPNCPIALLMINMEI': 'cpi',
                        'JPNRGDPQDSNAQ': 'gdp',
                        'JPNURNAA': 'urate',
                        'MYAGM2JPM189N': 'm2',
                        'INTGSTJPM193N': 'tbill'})

jf = jf.resample('A', how='mean').fillna(method='ffill').ix['1975':]
# jf.to_csv('/Users/tom/TradeData/DataStorage/inflation_recession/fred_japan.csv')
jf_n = jf.dropna() / jf.dropna().iloc[0]

#-----------------------------------------------------------------------------
# Plotting

jf.plot(subplots=True)
jf.ix['2007':].plot(subplots=True)

#-----------------------------------------------------------------------------
# IMF (Annual) Data
#-----------------------------------------------------------------------------

columns = {'Gross domestic product, constant prices': 'rgdp',
           'Gross domestic product, current prices': 'ngdp',
           'Gross domestic product, deflator': 'deflator',
           'Gross national savings': 'savings',
           'Inflation, average consumer prices': 'inflation',
           'Output gap in percent of potential GDP': 'gap',
           'Total investment': 'investment',
           'Unemployment rate': 'u_rate'
           }

df = pd.read_csv('/Volumes/HDD/Users/tom/DataStorage/inflation_recession/'
                 'weoreptc.aspx.xls', sep='\t', thousands=',')
df = parse_imf(df, column_formatter=columns)

j = df.xs('Japan', level='Country')
j.index = pd.DatetimeIndex(j.index)
j = j.join(jf[['tbill']].resample('AS', how='mean'))  # Pass list to keep name.
j['r_rate'] = j.tbill - j.inflation.pct_change() * 100
#-----------------------------------------------------------------------------
# Plotting

j.plot(subplots=True)

fig = plt.figure(figsize=(13, 8))
ax = fig.add_subplot(111)
df[['rgdp', 'u_rate']].plot(ax=ax)


(j.tbill - j.deflator.pct_change() * 100).plot(label='deflator', legend=True)
(j.tbill - j.inflation.pct_change() * 100).plot(legend=True, label='cpi')
plt.savefig('/Users/tom/Economics/inflation_in_recession/resources/'
            'japan_maybe_real_rate.png')

j[['r_rate', 'u_rate']].plot()
