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


def item_identifier(line, march=False, regex=None):
    if march:
        if line.startswith('D '):
            return True
        else:
            return None
    else:
        return id_monthly(line, regex)


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


def writer(dfs, date, store_path, march=False):
    store = pd.HDFStore(store_path + 'cps_store.h5')
    if march:
        hr_rec, fam_rec, pr_rec = dfs
        dct = {'house': hr_rec, 'family': fam_rec, 'person': pr_rec}
        for key, val in dct.iteritems():
            try:
                store.remove('/march_sup/dd/' + 'y' + date + key)
            except KeyError:
                pass
            store.append('/march_sup/dd/' + 'y' + date + key, val)
    else:
        pass

def main(file_, store_path='/Volumes/HDD/Users/tom/DataStorage/CPS/',
         march=False):
    if march:
        ids, lengths, starts = [], [], []
        year = file_[6:8]
        with open(file_, 'r') as f:
            for line in f:

                if item_identifier(line, march=True):
                    dict_constructor(line, ids, lengths, starts)
        df = pd.DataFrame({'length': lengths,
                           'start': starts,
                           'id': ids})
        dfs = checker(df)
        writer(dfs, year, store_path, march=True)
    else:
        with open(infile, 'r') as f:
            ids, lengths, starts, ends = [], [], [], []
            list_group = (ids, lengths, starts, ends)
            regex = make_reg()
            year = re.match(r'\w*(\d{2})', infile.split('/')[-1]).groups()[0]
            for line in f:
                grouped = month_item(line, reg=regex)
                if grouped:  # There is a match
                    month_dict_constructor(line, grouped, list_group)
        df = pd.DataFrame({'id': ids, 'length': lengths, 'start': starts,
                           'end': ends})
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


class Parser(object):

    def __init__(self, infile, outfile, regex=None):
        self.infile = infile
        self.outfile = outfile  # Full path for store.
        self.dataframes = []
        self.previous_line = None
        self.next_line = None
        self.holder = []
        if regex is None:
            self.regex = self.make_regex()
        self.pos_id = 0
        self.pos_len = 1
        self.pos_start = 2
        self.pos_end = 3
        self.store_name = self.get_store_name()  # name in Store.

    def run(self):
        with open(self.infile, 'r') as f:
            for line in f:
                if self.previous_line is None:
                    self.ids = []
                    self.lenghts = []
                    self.starts = []
                    self.ends = []
                self.analyze(line, f)
            # Finally

            to_be_df = self.holder
            df = pd.DataFrame(to_be_df, columns=['id', 'length', 'start',
                                                 'end'])
            df = pd.concat([self.common, df])
            self.dataframes.append(df)

    def analyze(self, line, f):
        maybe_groups = self.regex.match(line)
        if maybe_groups:
            formatted = self.formatter(maybe_groups)

            if len(self.holder) == 0:
                self.holder.append(formatted)
            # Fake break
            elif formatted[self.pos_start] > self.holder[-1][self.pos_end] + 1:
                self.handle_padding(formatted, f)
            # Real break
            elif formatted[self.pos_start] < self.holder[-1][self.pos_end]:
                self.handle_real_break(formatted)
            else:
                self.holder.append(formatted)

    def get_store_name(self):
        """
        #-----------------------------------------------------------------------
        NOTE: Usually, the documentation from January applies to an entire
        year. Execptions are 1984-1985 and 1994-1995. The January 1984
        documentation is used through to June 1985. The July 1985 documentation
        applies to the remainder of 1985. For 1994-1995, the January 1994
        documentation is used through August 1995. The September 1995
        documentation serves for the rest of the year.

        #-----------------------------------------------------------------------
        Not sure about jan2003 going till april 2004.


        We have {
                jan1989: [1989-01,1991-12], jan1992: [1992-01,1993-12],
                jan1994: [1994-01,1995-08], sep1995: [1995-09,1997-12],
                jan1998: [1998-01,2002-12], jan2003: [2003-01,2004-04],
                may2004: [2004-05,2005-07], aug2005: [2005-08,2006-12],
                jan2007: [2007-01,2008-12], jan2009: [2009-01,2009-12],
                jan2010: [2010-01,2012-04], may2012: [2012-05,2012-12]
                jan2013: [2013-01,2013-03]
                }
        """
        dict_ = {'cps89.ddf': 'jan1989',
                 'cps92.ddf': 'jan1992',
                 # MISSING 1994
                 'cpsbsep95.ddf': 'sep1995',
                 'cpsbjan98.ddf': 'jan1998',  # renamed this infile
                 'cpsbjan03.ddf': 'jan2003',
                 'cpsbmay04.ddf': 'may2004',
                 'cpsbaug05.ddf': 'aug2005',
                 'cpsbjan07.ddf': 'jan2007',
                 'cpsbjan09.ddf': 'jan2009',
                 'cpsbjan10.ddf': 'jan2010',
                 'cpsbmay12.ddf': 'may2012',
                 # Missing 2013. use alt parser
                 }
        inname = self.infile.split('/')[-1]
        store_name = dict_[inname]
        return store_name

    def handle_padding(self, formatted, f):
        """
        CPS left out some padding characters.

        Unpure.  Need to know next line to determine pad len.
        """
        # Can't use f.readline() cause final line would be infinite loop.

        # dr = it.dropwhile(lambda x: not self.regex.match(x), f)
        # next_line = next(dr)
        # maybe_groups = self.regex.match(next_line)

        # next_formatted = self.formatter(maybe_groups)
        last_formatted = self.holder[-1]
        pad_len = formatted[self.pos_start] - last_formatted[self.pos_end] - 1
        pad_str = last_formatted[self.pos_end] + 1
        pad_end = pad_len + pad_str - 1
        pad = ('PADDING', pad_len, pad_str, pad_end)

        self.holder.append(pad)
        self.holder.append(formatted)  # goto next line

    def handle_real_break(self, formatted):
        """
        CPS reuses some codes and then starts over.
        """
        to_be_df = self.holder
        df = pd.DataFrame(to_be_df, columns=['id', 'length', 'start',
                                             'end'])

        if len(self.dataframes) == 0:
            self.dataframes.append(df)
            common_index_pt = df[df['start'] == formatted[self.pos_end]].index[0] - 1
            self.common = df.ix[:common_index_pt]
        else:
            df = pd.concat([self.common, df], ignore_index=True)
            self.dataframes.append(df)

        self.holder = [formatted]  # next line

    def make_regex(self):
        return re.compile(r'(\w{1,2}[\$\-%]\w*|PADDING)\s*CHARACTER\*(\d{3})\s*\.{0,1}\s*\((\d*):(\d*)\).*')

    def formatter(self, match):
        """
        Conditional on a match, format them into a nice tuple of
            id, length, start, end
        """
        id_, length, start, end = match.groups()
        length = int(length)
        start = int(start)
        end = int(end)
        print(id_)
        return (id_, length, start, end)

    def len_checker(self):
        # Will fail cause CPS screwed up w/ padding.
        for df in self.dataframes:
            assert (df.end - df.start == df.length - 1).all()

    def con_checker(self):
        for df in self.dataframes:
            assert (df.end.shift() - df.start + 1 == 0)[1:].all()

    def writer(self):
        """
        Once you have all the dataframes, write them to that outfile,
        an HDFStore.
        """
        store = pd.HDFStore(self.outfile)
        for df in self.dataframes:
            try:
                store.remove('monthly/dd/' + self.store_name)
            except KeyError:
                pass
            store.append('monthly/dd/' + self.store_name, df)
        store.close()

if __name__ == '__main__':
    """
    Third arg must be True if you want to match a march dd.
    """

    import sys

    infile, outdir, march = sys.argv[1:]
    if march == 'True':
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
    #--------------------------------------------------------------------------
    else:
        kls = Parser(infile)
        kls.run()
        print('Added {}'.format(infile))
