import pandas as pd
from pandas.io.data import DataReader

# cd '/Volumes/HDD/Users/tom/DataStorage/Historical/Commodity Prices/'

xls = pd.ExcelFile('Cc205-266.xls')
xls.parse(xls.sheet_names[0])
df = xls.parse(xls.sheet_names[0], parse_dates='Year', index_col='Year')
df.columns = map(lambda x: '_'.join(x.split('_')[:2]), df.columns)

more_cols = ['IQ00000', 'M04F1AUS16980M260NNBR', 'CP0118PLM086NEST',
             'CUUR0000SEFR', 'ID52']

more = pd.concat([DataReader(x, data_source='fred', start='2007-01-01')
                  for x in more_cols], axis=1)

moremore = ['DGS5', 'DFII5']
