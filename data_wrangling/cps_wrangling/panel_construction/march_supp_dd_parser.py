"""
Should be useable for some years.

A few different styles:

March Supplement Style
======================

>    Data lines should be formatted like (separated by variable whitespace)
>
>    D Abbreviation length start_position (valid_low:valid_high)
>
>        DESCRIPTION
#-----------------------------------------------------------------------------


89, 92, ... style
=================

Header starts with `*`

DATA         SIZE               BEGIN:END

e.g.

    H$CPSCHK    CHARACTER*001 .     (0001:0001)           ALL

so either [H,HG,L,A,C,M][$-%]|PADDING

"""
import re
import itertools as it

import ipdb
import pandas as pd


def item_identifier(line, regex=None):
    if line.startswith('D '):
        return True
    else:
        return None


def id_monthly(line, regex):
    if regex.match(line):
        return True


def item_parser(line):
    return line.split()


def dict_constructor(line, ids, lengths, starts):
    _, id_, length, start = item_parser(line)[:4]
    ids.append(id_)
    lengths.append(int(length))
    starts.append(int(start))


def checker(df):
    check = df.start - df.length.cumsum().shift() - 1
    # ignore the first break.  That's the start.
    breaks = check[check.diff() != 0].dropna().index[1:]
    if len(breaks) == 0:
        return [df]
    dfs = [df.ix[:x - 1] for x in breaks]
    dfs.append(df.ix[breaks[-1]:])
    return dfs


def writer(dfs, date, store_path):
    store = pd.HDFStore(store_path + 'cps_store.h5')

    hr_rec, fam_rec, pr_rec = dfs
    dct = {'house': hr_rec, 'family': fam_rec, 'person': pr_rec}
    for key, val in dct.iteritems():
        try:
            store.remove('/march_sup/dd/' + 'y' + date + key)
        except KeyError:
            pass
        store.append('/march_sup/dd/' + 'y' + date + key, val)


def main(file_, store_path='/Volumes/HDD/Users/tom/DataStorage/CPS/'):

    ids, lengths, starts = [], [], []
    year = file_[6:8]
    with open(file_, 'r') as f:
        for line in f:

            if item_identifier(line):
                dict_constructor(line, ids, lengths, starts)
    df = pd.DataFrame({'length': lengths,
                       'start': starts,
                       'id': ids})
    dfs = checker(df)
    writer(dfs, year, store_path)


def month_dict_constructor(line, grouped, list_group):
    """
    Take a line and that line parsed and some lists and append the
    values to appropriate list.  Very unpure
    """
    id_, length, start, end = grouped.groups()
    ids, lengths, starts, ends = list_group
    ids.append(id_)
    lengths.append(int(length))
    starts.append(int(start))
    ends.append(int(end))
    return list_group


def month_item(string, reg=None):
    if reg is None:
        reg = make_reg()
    return reg.match(string)


def make_reg():
    return re.compile(r'(\w{1,2}[\$\-%]\w*|PADDING)\s*CHARACTER\*(\d{3})\s*\.\s*\((\d*):(\d*)\).*')


def padder(df):
    """
    Some places missing padding.
    e.g. [31-37] of 89.
    """
    diffed = df.end.shift() - df.start + 1
    breaks = diffed[diffed != 0].dropna()
    padders = []
    padders_ind = breaks[breaks < 0]
    real_breaks = breaks[breaks > 0]
    for breaker in padders_ind:
        start = df.ix[abs(breaker) - 1]['end']
        length = int(abs(breaker))
        id_ = 'PADDING'
        end = start + length
        padders.append((id_, length, start, end))
    #--------------------------------------------------------------------------
    # Pickup the real breaks
    n_dfs = len(real_breaks) + 1
    real_break_points = rea_breaks.index
    for breaker in real_break_points:
        len_break = len(breaks.ix[:breaker - 1])
        this_pad = padders[:len_break]

    # Pickup the missing pads
    add_on = pd.DataFrame(padders,
                          columns=['id', 'length', 'start', 'end'])
    padded_df = pd.concat([df, add_on]).sort([])


def make_padder(breaker):
    start = df.ix[breaker - 1].end
    length = abs(diffed.ix[breaker])
    id_ = 'PADDER'
    end = start + length

if __name__ == '__main__':
    """

    """

    import sys

    infile, outdir = sys.argv[1:]

    years1 = [str(x) for x in range(64, 100)]
    years2 = ['0' + str(x) for x in range(1, 10)]
    years3 = ['10', '11', '12']
    years = years1 + years2 + years3
    #--------------------------------------------------------------------------
    for yr in years:
        infile = 'cpsmar' + yr + '.ddf'
        try:
            main(infile)
            print('Added {}'.format(infile))
        except IOError:
            print('No File for {}'.format(infile))
