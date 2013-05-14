# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

plt.matplotlib.matplotlib_fname()

# <codecell>

import sys
sys.path.append('/Users/tom/bin')

# <codecell>

import numpy as np
import pandas as pd
from pandas.io.data import DataReader

from parse_imf import imf_process

# <codecell>

series = ['DGS10', 'CPIAUCSL', 'UNRATE']
df1, df2, df3 = [DataReader(x, data_source='fred', start='1-1-1940') for x in series]
base = '/Users/tom/Economics/inflation_in_recession/resources/'

# <codecell>

df = df1.join(df2.join(df3))

# <codecell>

def add_rec_bars(ax, dates=None):
    if dates is None:
        dates = pd.read_csv('/Users/tom/bin/rec_dates.csv', parse_dates=['Peak', 'Trough'])
    for row in dates.iterrows():
        x = row[1]
        y1, y2 = ax.get_ylim()
        ax.fill_between(x, y1=y1, y2=y2, alpha=.25, color='k')
    return ax

# <codecell>

def f(x):
    try:
        return float(x)
    except:
        return np.nan

# <codecell>

df['DGS10'] = df.DGS10.apply(f)

# <codecell>

df.ix['2012-02'].head()

# <codecell>

df = df.resample(rule='M', how='mean')

# <codecell>

df.ix['2012'].head()

# <codecell>

df = df.fillna(method='ffill')
#df['CPIAUCSL'] = df['CPIAUCSL'].pct_change()

# <codecell>

plt.figsize(16, 9)
ax = df.plot(secondary_y='CPIAUCSL')
ax = add_rec_bars(ax)
plt.savefig(base + 'all_cpi_unemployment_treasury.png', dpi=300)

# <codecell>

ax = df.ix['2006':].plot(secondary_y='CPIAUCSL')
ax = add_rec_bars(ax)
plt.savefig(base + 'gr_cpi_unemployment_treasury.png', dpi=300)

# <codecell>

df_pc = df.copy()
df_pc['CPIAUCSL'] = df_pc['CPIAUCSL'].pct_change(4) * 100
ax = df_pc.ix['2006':].plot()
ax = add_rec_bars(ax)
plt.savefig(base + 'gr_4q_pc.png', dpi=300)

# <codecell>

df_pc['ma4_CPIAUCSL'] = pd.rolling_mean(df_pc.CPIAUCSL, 4)
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1 = df_pc[['DGS10', 'ma4_CPIAUCSL', 'UNRATE']].plot(secondary_y=['ma4_CPIAUCSL'], ax=ax1)
ax1 = add_rec_bars(ax1)
ax2 = fig.add_subplot(2, 1, 2)
ax2 = df_pc.ix['2006':][['DGS10', 'UNRATE', 'ma4_CPIAUCSL']].plot(secondary_y=['ma4_CPIAUCSL'], ax=ax2)
ax2 = ax1 = add_rec_bars(ax2)
plt.savefig(base + '/both_ma.png', dpi=300)

# <headingcell level=2>

# Japaneese Data

# <codecell>

series = ['JPNCPIALLMINMEI',  # CPI for all items 
          'JPNRGDPQDSNAQ',    # nominal GDP
          'JPNURAMS',         # Adjusted Unemployment Rate
          'MYAGM2JPM189N',    # M2
          'INTGSTJPM193N'     # T-bill rate
          ]

# <codecell>

cpi, gdp, urate, m2, tbill = [DataReader(x, data_source='fred', start='1-1-1940') for x in series]
tbill = tbill.applymap(f)

# <codecell>

jf = cpi.join(urate.join(m2.join(tbill.join(gdp))))
jf = jf.rename(columns={'JPNCPIALLMINMEI' : 'cpi',
                        'JPNRGDPQDSNAQ' : 'gdp',
                        'JPNURAMS' : 'urate',
                        'MYAGM2JPM189N': 'm2',
                        'INTGSTJPM193N': 'tbill'})

# <codecell>

cpi.join(urate.join(m2.join(tbill))).ix['2010']

# <codecell>

print(gdp.ix['2010'])

# <codecell>

print(tbill.ix['2010'])
print(jf.ix['2010'])

# <codecell>

tbill.ix['2010']

# <codecell>

jf = jf.resample('M', how='mean').fillna(method='ffill').ix['1975':]

# <codecell>

jf['cpi'].plot()

# <codecell>

jf[['cpi', 'urate', 'm2']].plot()

# <headingcell level=2>

# IMF (Annual) Data

# <codecell>

columns={'Gross domestic product, constant prices': 'rgdp',
         'Gross domestic product, current prices': 'ngdp',
         'Gross domestic product, deflator': 'deflator',
         'Gross national savings': 'savings',
         'Inflation, average consumer prices': 'inflation',
         'Output gap in percent of potential GDP': 'gap',
         'Total investment': 'investment',
         'Unemployment rate': 'u_rate'
         }

df = pd.read_csv('/Users/tom/Desktop/weoreptc.aspx.xls', sep='\t', thousands=',')

# <codecell>

df['1980']

# <codecell>

df = imf_process(df, column_formatter=columns)

# <codecell>

df[['rgdp', 'u_rate']].groupby(level='Country').plot()

# <codecell>

fig = plt.figure(figsize=(13, 8))
ax = fig.add_subplot(111)
df[['rgdp', 'u_rate']].plot(ax=ax)

# <codecell>


