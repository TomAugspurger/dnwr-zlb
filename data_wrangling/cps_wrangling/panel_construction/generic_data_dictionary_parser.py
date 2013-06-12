import re
import itertools as it

import pandas as pd
import ipdb


class Parser(object):
    """
    Handles most years.

    Parameters
    ----------
    infile: A path to the data dictionary
    outfile: path to HDFStore holding output.
    style: Optional mmmYYYY for regex.
    regex: Not used I think.
    """

    def __init__(self, infile, outfile, style=None, regex=None):
        self.infile = infile
        self.outfile = outfile  # Full path for store.
        self.dataframes = []
        # self.previous_line = None
        self.next_line = None
        self.holder = []
        if regex is None:
            self.regex = self.make_regex(style=style)
        self.style = style
        self.pos_id = 0
        self.pos_len = 1
        self.pos_start = 2
        self.pos_end = 3
        self.store_name = self.get_store_name()  # name in Store.

    def run(self):
        with open(self.infile, 'r') as f:
            for line in f:
                # if self.previous_line is None:
                    # self.ids = []
                    # self.lenghts = []
                    # self.starts = []
                    # self.ends = []
                self.analyze(line.rstrip(), f)

            # Finally
            to_be_df = self.holder
            df = pd.DataFrame(to_be_df, columns=['id', 'length', 'start',
                                                 'end'])
            # Some years need to grab the very last one
            # If there's only one, it's been picked up.
            if len(self.dataframes) > 0:
                df = pd.concat([self.common, df])
            self.dataframes.append(df)

    def analyze(self, line, f):
        maybe_groups = self.regex.match(line)

        if maybe_groups:
            formatted = self.formatter(maybe_groups)
            # Return to main for loop under run
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
                jan1989: [1989-01, 1991-12], chk
                jan1992: [1992-01, 1993-12], chk
                jan1994: [1994-01, 1994-03], chk
                apr1994: [1994-04, 1995-05], chk
                jun1995: [1995-06, 1995-08], chk
                sep1995: [1995-09, 1997-12], chk
                jan1998: [1998-01, 2002-12], chk
                jan2003: [2003-01, 2004-04], chk
                may2004: [2004-05, 2005-07], chk
                aug2005: [2005-08, 2006-12], chk
                jan2007: [2007-01, 2008-12], chk
                jan2009: [2009-01, 2009-12], chk
                jan2010: [2010-01, 2012-04], boom #dat
                may2012: [2012-05, 2012-12], 
                jan2013: [2013-01, 2013-03],
                }
        """
        dict_ = {'cps89.ddf': 'jan1989',
                 'cps92.ddf': 'jan1992',
                 'cpsb_apr94_may95.ddf': 'apr1994',
                 'cpsb_jan94_mar94.ddf': 'jan1994',
                 'cpsb_jun95_aug95.ddf': 'jun1994',
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

    def make_regex(self, style=None):
        if style is None:
            return re.compile(r'(\w{1,2}[\$\-%]\w*|PADDING)\s*CHARACTER\*(\d{3})\s*\.{0,1}\s*\((\d*):(\d*)\).*')
        elif style is 'aug2005':
            return re.compile(r'(\w+)[\s\t]*(\d{1,2})[\s\t]*(.*?)[\s\t]*\(*(\d+)\s*-\s*(\d+)\)*$')
        elif style is 'jan1998':
            return re.compile(r'D (\w+) \s* (\d{1,2}) \s* (\d*)')

    def formatter(self, match):
        """
        Conditional on a match, format them into a nice tuple of
            id, length, start, end

        match is a regex object.
        """
        if self.style == 'jan1998':
            id_, length, start = match.groups()
            length = int(length)
            start = int(start)
            end = start + length - 1
        else:
            try:
                id_, length, start, end = match.groups()
            except ValueError:
                id_, length, description, start, end = match.groups()
            length = int(length)
            start = int(start)
            end = int(end)
        return (id_, length, start, end)

    def len_checker(self):
        # Will fail cause CPS screwed up w/ padding.
        for df in self.dataframes:
            assert (df.end - df.start == df.length - 1).all()

    def con_checker(self):
        for df in self.dataframes:
            if not (df.end.shift() - df.start + 1 == 0)[1:].all():
                badlines = df[~(df.end.shift() - df.start + 1 == 0)][1:].index
                for line in badlines:
                    print(df.ix[line - 1:line + 1])
                raise ValueError('Continuity is Broken Around Here')

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
