from __future__ import unicode_literals

import re

import pandas as pd

# #-----------------------------------------------------------------------------
# col_widths = [15, 2, 4, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 10, 2, 2, 2, 2,
#               2, 2, 2, 5, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 5, 3, 1, 1, 1, 1, 3,
#               3, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#               2, 2, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2,
#               2, 2, 2, 3, 2, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2,
#               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#               2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2,
#               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6, 2, 2, 6, 2, 2, 2, 2, 2, 2, 2,
#               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#               4, 4, 4, 4, 1, 2, 8, 1, 4, 8, 8, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#               2, 2, 2, 2, 10, 10, 10, 10, 10, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#               2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 2, 2, 2,
#               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 10, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2,
#               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#               2, 2, 2, 2, 2, 15]
# #-----------------------------------------------------------------------------
# gen = pd.read_fwf('apr13pub.dat', widths=col_widths, chunksize=960)
# df = pd.concat(x for x in gen)

# with pd.get_store('/Volumes/HDD/Users/tom/DataStorage/CPS/apr_2013.h5') as f:
#     f.append('apr_2013', df)
#-----------------------------------------------------------------------------
# Interesting Columns
# names = ['HRHHID', 'HRMONTH', 'HRYEAR4', 'HRMIS']
# positions = [0, 1, 2, 19]

#-----------------------------------------------------------------------------
# Dictionary Parsing
r = re.compile(ur"(\w+) \s* (\d+) .* (\d+\s*-\s*\d+)\s*$")
# Match a word followed by spaces, then digits (col width), followed by
# anything, followed by digits - digits (col locations).
# For whatever reason they'll occasionally use a unicode '-'
# with byes '\xe2\x80\x93'.
f = open('/Volumes/HDD/Users/tom/DataStorage/CPS/jan13dd.txt')

matches = []
for line in f:
    line = line.decode('utf-8')
    line = re.sub(ur'\u2013', '-', line)
    m = r.match(line)
    if m:
        matches.append(m.groups())


def pos_checker(matches):
    warns = []
    for i, tup in enumerate(matches):
        if i == 0:
            low, high = [x.strip() for x in tup[-1].split('-')]
            continue
        next_l, next_h = [x.strip() for x in tup[-1].split('-')]
        if int(next_l) - int(high) > 1:
            warns.append((i, next_l, next_h))
        low, high = next_l, next_h
    return warns

warns = pos_checker(matches)

with open('jan13_dd_parsed.txt', 'wt') as f:
    for line in matches:
        f.write(','.join(x for x in line) + '\n')
    print('Wrote the parsed file.')

df = pd.read_csv('/Volumes/HDD/Users/tom/DataStorage/CPS/jan13_dd_parsed.txt',
                 sep=r',|\s*-\s*', names=['id', 'length', 'start', 'stop'])
store = pd.HDFStore('/Volumes/HDD/Users/tom/DataStorage/CPS/cps_store.h5')
store_name = 'jan2013'
try:
    store.remove('monthly/dd/' + store_name)
except KeyError:
    pass
store.append('monthly/dd/' + store_name, df)
print('Wrote the dd.')
store.close()
