"""
Download the monthly files for 1976 - 2013 (March).

#-----------------------------------------------------------------------------

Match on:
http://www.nber.org/cps-basic/cpsb****.Z // ** is [7601, 0912]
and then ['jan10', ... 'mar13']

Change the variable out_dir as necessary for your machine's filesystem.

http://www.nber.org/data/cps_basic.html
"""
# TODO: Get list of files from from settings. Currently just grabs all new.

from functools import partial
from itertools import ifilter
import re
import os
from os.path import exists
import urllib2

from lxml.html import parse

from renamer import renamer


def matcher(link, regex):
    try:
        _, fldr, file_ = link[2].split('/')
        if regex.match(file_):
            return file_
    except ValueError:
        pass


def downloader(link, out_dir, dl_base="http://www.nber.org/cps-basic/"):
    """
    Link is a str like cpsmar06.zip; It is both the end of the url
    and the filename to be used.
    """
    content = urllib2.urlopen(dl_base + link)
    with open(out_dir + link, 'w') as f:
        f.write(content.read())


def _exists(path_name):
    no_ext = path_name.split('.')[0]
    if exists(path_name) or exists(no_ext + '.gz') or exists(no_ext + '.gz'):
        return True


def main(out_dir, regex):
    site = 'http://www.nber.org/data/cps_basic.html'
    parsed = parse(urllib2.urlopen(site))
    root = parsed.getroot()

    partial_matcher = partial(matcher, regex=regex)

    for link in ifilter(partial_matcher, root.iterlinks()):
        _, _, _fname, _ = link
        fname = _fname.split('/')[-1]
        existing = _exists(os.path.join(out_dir, fname))
        if not existing:
            downloader(fname, out_dir)
            print('Added {}'.format(fname))

    renamer(out_dir)

if __name__ == '__main__':
    """
    Use '.zip' for the datafiles themselves,
        '.ddf' for the data dictionaries, or
        '.pdf' for the pdf documentation.

    example outdir: '/Volumes/HDD/Users/tom/DataStorage/CPS/monthly/'
    """
    import json
    import sys
    with open('settings.txt') as f:
        settings = json.load(f)

    out_dir = settings["raw_monthly_path"]

    try:
        dd = sys.argv[1]
    except IndexError:
        dd = False

    if dd == 'dd':  # download the data dictionaries
        regex = re.compile(r'\w*\.ddf')
    else:
        regex = re.compile(r'cpsb\d{4}.Z|\w{3}\d{2}pub.zip|\.[ddf,asc]$')
    main(out_dir, regex)
