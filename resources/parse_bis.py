import pandas as pd
from date_parsers import year_month_bis

df = pd.read_csv('sec1.csv', index_col=0)
s = df.dropna(how='all').stack()
s = s.reset_index()


def change_dates(x):
    try:
        return year_month_bis(x)
        print(x)
    except KeyError:
        return None

s = s.iloc[1:]
s['dates'] = s.level_1.apply(lambda x: change_dates(x))
df = df.drop('level_1')
df.rename(columns={0: 'value'})
