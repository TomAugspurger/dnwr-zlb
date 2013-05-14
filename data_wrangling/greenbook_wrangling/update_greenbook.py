import numpy as np
import pandas as pd

from date_parsers import coibon_greenbook
#-----------------------------------------------------------------------------
# IO
xls = pd.ExcelFile('GreenBookForecasts_for_AER.xlsx')
df = xls.parse(xls.sheet_names[0])
#-----------------------------------------------------------------------------
# Index Formatting
s = pd.Series(df.columns.tolist())  # First row of column MultiIndex
cols2 = df.iloc[0]  # Second row of column MultiIndex
s = s.str.replace(r'(Unnamed: \d+)', '_fill_me_')
s = s.replace('_fill_me_', np.nan)
s = s.fillna(method='ffill')
cols1 = s.tolist()
cols2 = cols2.fillna('FFR')
df.columns = pd.MultiIndex.from_tuples(zip(cols1, cols2))
df = df.iloc[1:]

idx = [coibon_greenbook(x) for x in df.index]
df.index = pd.DatetimeIndex(idx)
df = df.astype('float')
