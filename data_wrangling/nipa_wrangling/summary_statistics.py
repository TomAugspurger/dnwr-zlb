import pandas as pd

path = '/Volumes/HDD/Users/tom/DataStorage/nipa/'
dta = pd.read_csv(path + 'cleaned_nipa_gdp_1947-2013.csv', index_col=0).ix[:'2011-04']
