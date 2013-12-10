"""
Running of the regressions
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm

from helpers import add_demo_dummies

with pd.get_store('/Volumes/HDD/Users/tom/DataStorage/CPS/analyzed/clean.h5') as store:
    df = store.select('cleaned')

s_dummies = pd.get_dummies(df.quarter).rename(columns={1: "Q1", 2: "Q2",
                                                       3: "Q3", 4: "Q4"})
df = pd.concat([df, s_dummies], axis=1)

# Add time trend
df['trend'] = 4 * (df.year - 1996) + df.quarter

# Filter out outer .005% of hours and real hourly earnings
low_hours, high_hours = df.hours.quantile(.005), df.hours.quantile(.995)
low_earn, high_earn = df.real_hr_earns.quantile(.003), df.real_hr_earns.quantile(.997)

trim = df.query('hours > low_hours & hours < high_hours &'
                'real_hr_earns > low_earn & real_hr_earns < high_earn')

# Deseasonalize real hourly wages

seasonal_mod = sm.OLS.from_formula("np.log(real_hr_earns) ~ Q1 + Q2 + Q3", trim)
seasonal_res = seasonal_mod.fit()

seasonal_effects = seasonal_res.params
seasonal_effects['Q4'] = 0

# BUG https://github.com/pydata/pandas/issues/5653
# real_hr_earns_d = df.set_index('quarter', append=True)['real_hr_earns'].copy()
# real_hr_earns_d = real_hr_earns_d.rename(index={1: "Q1", 2: "Q2", 3: "Q3",
#                                                 4: "Q4"}, level='quarter')

real_hr_earns_d = trim[['quarter', 'real_hr_earns']].copy()
real_hr_earns_d['quarter'] = real_hr_earns_d['quarter'].replace({1: "Q1", 2: "Q2",
                                                                 3: "Q3", 4: "Q4"})
real_hr_earns_d = real_hr_earns_d.set_index('quarter', append=True)
# Series for sub
real_hr_earns_d = np.log(real_hr_earns_d['real_hr_earns']).sub(seasonal_effects,
                                                               level='quarter')
idx = real_hr_earns_d.index.droplevel('quarter')  # to match with trim
real_hr_earns_d.index = idx
trim['ln_real_hr_earns_d'] = real_hr_earns_d


trim = add_demo_dummies(trim)
trim['expr_2'] = trim.expr ** 2
trim['expr_3'] = trim.expr ** 3
trim['expr_4'] = trim.expr ** 4

trim['og_weight'] = trim.og_weight / 10000  # do earleir.
trim = trim[~pd.isnull(trim.either_history.replace(-1, np.nan))]
with pd.get_store('/Volumes/HDD/Users/tom/DataStorage/CPS/analyzed/clean.h5') as store:
    trim.to_hdf(store, 'trim', format='f', append=False)

# First Stage

mod_fs = sm.OLS.from_formula("ln_real_hr_earns_d ~ age + sex_d + race_d +"
                             "married_d + edu_bin + expr + expr_2 +"
                             "expr_3 + expr_4 + trend", trim)
res_fs = mod_fs.fit()
#-----------------------------------------------------------------------------
# Second Stage
trim['wage_index'] = res_fs.resid
g = trim.reset_index().groupby(['qmonth', 'either_history'])
mean_ln_earn_ts = g['ln_real_hr_earns_d'].mean().unstack()[[0, 1]].rename(columns={0: 'from_e',
                                                                                   1: 'from_n'})
mean_ln_earn_ts_d = mean_ln_earn_ts.diff()

with pd.get_store('/Volumes/HDD/Users/tom/DataStorage/CPS/analyzed/analyzed_store.h5') as store:
    prod = store.select('bls_productivity_compensation')['productivity']

prod_ln_d = pd.DataFrame(np.log(prod).diff().dropna())
df_ss = pd.concat([mean_ln_earn_ts_d, prod_ln_d], axis=1).dropna()

mod_from_e = sm.OLS.from_formula("from_e ~ productivity", df_ss)
res_from_e = mod_from_e.fit()

mod_from_n = sm.OLS.from_formula("from_n ~ productivity", df_ss)
res_from_n = mod_from_n.fit()


# Second Stage medians

median_ln_earn_ts = g['ln_real_hr_earns_d'].median().unstack()[[0, 1]].rename(columns={0: 'from_e',
                                                                                       1: 'from_n'})
median_ln_earn_ts_d = median_ln_earn_ts.diff()

df_ss = pd.concat([median_ln_earn_ts_d, prod_ln_d], axis=1).dropna()

mod_from_e = sm.OLS.from_formula("from_e ~ productivity", df_ss)
res_from_e = mod_from_e.fit()

mod_from_n = sm.OLS.from_formula("from_n ~ productivity", df_ss)
res_from_n = mod_from_n.fit()


#-----------------------------------------------------------------------------
# Using weights
mod_fs = sm.OLS.from_formula("ln_real_hr_earns_d ~ age + sex_d + race_d +"
                             "married_d + edu_bin + expr + expr_2 +"
                             "expr_3 + expr_4 + trend", trim)
res_fs = mod_fs.fit()
trim['wage_index'] = res_fs.resid

weights = trim.og_weight.div(trim.og_weight.groupby(level='qmonth').sum(),
                             level='qmonth')

trim['wage_index_weighted'] = trim.wage_index.mul(weights, level='qmonth')
g = trim.reset_index().groupby(['qmonth', 'either_history'])
res = g['wage_index_weighted'].sum()


#-----------------------------------------------------------------------------
# By major sector (see notes.md)

durable = trim[trim.industry.isin([5, 6, 7, 8, 9, 10, 11, 12, 13])]
nondurable = trim[trim.industry.isin([14, 15, 16, 17, 18, 19, 20])]

gd = durable.reset_index().groupby(['qmonth', 'either_history'])
mean_ln_earn_ts = gd['ln_real_hr_earns_d'].mean().unstack()[[0, 1]].rename(columns={0: 'from_e',
                                                                                    1: 'from_n'})
mean_ln_earn_ts_d = mean_ln_earn_ts.diff()

with pd.get_store('/Volumes/HDD/Users/tom/DataStorage/CPS/analyzed/analyzed_store.h5') as store:
    prod = store.select('major_sectors_output_per_hour')

mean_ln_earn_ts_d['prod'] = prod['durable'].diff()
