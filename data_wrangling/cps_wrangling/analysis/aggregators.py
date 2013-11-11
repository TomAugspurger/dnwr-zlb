import json

import arrow
import pandas as pd

from helpers import (get_from_earn, get_earn_quantiles, get_earn_summary,
                     get_earn_diff_with_flow, make_MultiIndex_date)


with open('../panel_construction/settings.txt', 'r') as f:
    settings = json.load(f)

analyzed_store = pd.HDFStore(settings['analyzed_path'])

earn_store = pd.HDFStore(settings['earn_store_path'])
panel_store = pd.HDFStore(settings['panel_store_path'])
cps_store = pd.HDFStore(settings['store_path'])

data_dir = settings['base_path']

earn_store = pd.HDFStore

m0 = arrow.get('1996-12', 'YYYY-MM')
# mn = arrow.get(earn_store.keys()[-1], '/mYY_MM')  # slow
mn = arrow.get('/m2013_09', '/mYY_MM')
months = [x.strftime('%Y_%m') for x in arrow.Arrow.range('month', m0, mn)]
medians = [get_from_earn(x, earn_store) for x in months]

medians = pd.Series(medians, index=pd.to_datetime(months, format='%Y_%m'))
# medians.to_csv('/Volumes/HDD/Users/tom/DataStorage/CPS/analyzed/median_wages.csv')
medians.to_hdf(analyzed_store, 'wages_median')

quants = {month: get_earn_quantiles(month, earn_store) for month in months}
quantiles = pd.DataFrame(quants).T
quantiles.index = pd.to_datetime(quantiles.index, format='%Y_%m')
# quantiles.to_csv('/Volumes/HDD/Users/tom/DataStorage/CPS/analyzed/wage_change_quantiles.csv')
quantiles.to_hdf(analyzed_store, 'wage_change_quantiles')

summary = {month: get_earn_summary(month, earn_store) for month in months}
summaries = pd.DataFrame(summary).T
summaries.index = pd.to_datetime(summaries.index, format='%Y_%m')
# summaries.to_csv('/Volumes/HDD/Users/tom/DataStorage/CPS/analyzed/wage_change_summaries.csv')
summaries.to_hdf(analyzed_store, 'wage_change_summariies')

diff_with_flow = {month: get_earn_diff_with_flow(month, earn_store) for month in months}
diff_with_flow = {month: make_MultiIndex_date(df, month).dropna(how='all') for month, df
                  in diff_with_flow.iteritems()}
df = pd.concat(diff_with_flow.values())
df = df.sort_index()
df.loc[df[df.flow.isin(['ne', 'nu', 'nn', 'ue', 'uu', 'un'])].index, 4] = 0
df.loc[df[df.flow.isin(['en', 'un', 'nn', 'eu', 'uu', 'nu'])].index, 8] = 0
df = df.dropna()  # left with only clean values
# df.to_csv('/Volumes/HDD/Users/tom/DataStorage/CPS/analyzed/diff_with_flow.csv')
df.to_hdf(analyzed_store, 'wage_change_with_flow')


# hasn't been written yet.d
# group_summaries = {month: get_earn_summary_group(month, earn_store) for month in months}
