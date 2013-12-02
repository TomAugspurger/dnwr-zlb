import json

import pandas as pd
import pandas.io.data as web

import statsmodels.api as sm

# Nonfarm Business Sector: Real Output Per Hour of All Persons (OPHNFB)

with open('../cps_wrangling/panel_construction/settings.txt', 'r') as f:
    settings = json.load(f)

analyzed = pd.HDFStore(settings['analyzed_path'])

#-----------------------------------------------------------------------------
# productivity
df = web.DataReader("OPHNFB", data_source="fred", start="1950", end="2013-09")
df.to_hdf(analyzed, 'productivity')

#-----------------------------------------------------------------------------
# http://research.stlouisfed.org/fred2/series/PCEPI

df = web.DataReader("PCEPI", data_source="fred", start='1950', end='2013-09')
df.to_hdf(analyzed, 'pce_price_index', format='table')

#-----------------------------------------------------------------------------
# Unemployment
# http://research.stlouisfed.org/fred2/series/UNRATE
df = web.DataReader("UNRATE", data_source="fred", start='1950', end='2013-09')
cycle, trend = sm.tsa.filters.hpfilter(df, 129600)
df['cycle'] = cycle
df['trend'] = trend
df['shifted'] = df['cycle'] + df['trend']
df.to_hdf(analyzed, 'un_rate', format='table', append=False)

#-----------------------------------------------------------------------------
# Jolts
# Job Openings: Total Nonfarm (JTSJOL)
# Hires: Total Nonfarm (JTSHIL)
# Total Separations: Total Nonfarm (JTSTSL)
# Quits: Total Nonfarm (JTSQUR)
# Layoffs and Discharges: Total Nonfarm (JTSLDL)
series = ["JTSJOL", "JTSHIL", "JTSTSL", "JTSQUR", "JTSLDL"]
df = pd.concat([web.DataReader(x, data_source="fred", start='2000',
                               end='2013-09') for x in series], axis=1)
df.columns = ['openings', 'hires', 'separations', 'quits', 'layoffs']
df.to_hdf(analyzed, 'jolts', format='table', append=False)
