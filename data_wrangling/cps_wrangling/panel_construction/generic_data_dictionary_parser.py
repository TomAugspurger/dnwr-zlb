"""
Should be useable for some years.

Data lines should be formatted like (separated by variable whitespace)

D Abbreviation length start_position (valid_low:valid_high)

    DESCRIPTION
"""
import ipdb
import pandas as pd


def item_identifier(line):
    if line.startswith('D '):
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
    dfs = [df.ix[:x - 1] for x in breaks]
    dfs.append(df.ix[breaks[-1]:])
    return dfs


def writer(dfs, year, store_path):
    store = pd.HDFStore(store_path + 'cps_store.h5')
    hr_rec, fam_rec, pr_rec = dfs
    dct = {'house': hr_rec, 'family': fam_rec, 'person': pr_rec}
    for key, val in dct.iteritems():
        try:
            store.remove('/march_sup/dd/' + 'y' + year + key)
        except KeyError:
            pass
        store.append('/march_sup/dd/' + 'y' + year + key, val)


def main(file_, store_path='/Volumes/HDD/Users/tom/DataStorage/CPS/'):
    ids, lengths, starts = [], [], []
    year = file_[6:8]
    with open(file_, 'r') as f:
        for line in f:
            # ipdb.set_trace()
            if item_identifier(line):
                dict_constructor(line, ids, lengths, starts)
    df = pd.DataFrame({'length': lengths,
                       'start': starts,
                       'id': ids})
    dfs = checker(df)
    writer(dfs, year, store_path)

if __name__ == '__main__':
    years1 = [str(x) for x in range(64, 100)]
    years2 = ['0' + str(x) for x in range(1, 10)]
    years3 = ['10', '11', '12']
    years = years1 + years2 + years3
    #--------------------------------------------------------------------------
    for yr in years:
        fname = 'cpsmar' + yr + '.ddf'
        try:
            main(fname)
            print('Added {}'.format(fname))
        except IOError:
            print('No File for {}'.format(fname))
