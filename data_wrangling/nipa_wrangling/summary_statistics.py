import json

import pandas as pd
import statsmodels.tsa.api as sm

with open('settings.json') as f:
    settings = json.load(f)

store_path, store_name = settings['store_path'], settings['store_name']

with pd.get_store(store_path) as store:
    df = store.get(store_name)

#-----------------------------------------------------------------------------
df.fed_funds.update(df.tbill)
cols = ['real_pce', 'gdp', 'gdp_deflator', 'fed_funds']

res = (sm.filters.hpfilter(df[x]) for x in cols)
stds = ((x.std(), y.std()) for x, y in res)
