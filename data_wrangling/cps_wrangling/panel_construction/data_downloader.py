"""
Download the March Supplements for 1998 - 2012 and the data dictionaries
for available years.


#-----------------------------------------------------------------------------
Using http://www.nber.org/data/current-population-survey-data.html

Match on:
http://www.nber.org/cps/cpsmar**.zip // ** is [62, ... 99, 00, ... 12]

Change the variable out_dir as necessary for your machine's filesystem.
"""
import urllib2
from lxml.html import parse


def matcher(link, ftype):
    try:
        _, fldr, file_ = link[2].split('/')
        if file_.startswith('cpsmar') and file_.endswith(ftype):
            return file_
    except ValueError:
        return None


def downloader(link, out_dir, dl_base="http://www.nber.org/cps/"):
    """
    Link is a str like cpsmar06.zip; It is both the end of the url
    and the filename to be used.
    """
    content = urllib2.urlopen(dl_base + link)
    with open(out_dir + link, 'w') as f:
        f.write(content.read())


def main(out_dir, ftype):
    site = 'http://www.nber.org/data/current-population-survey-data.html'
    parsed = parse(urllib2.urlopen(site))
    root = parsed.getroot()
    for link in root.iterlinks():
        fname = matcher(link, ftype)
        if fname:
            downloader(fname, out_dir)
            print('Added {}'.format(fname))

if __name__ == '__main__':
    """
    Use '.zip' for the datafiles themselves,
        '.ddf' for the data dictionaries, or
        '.pdf' for the pdf documentation.

    example outdir: '/Volumes/HDD/Users/tom/DataStorage/CPS/march_supplements/'
    """
    import sys
    out_dir, ftype = sys.argv[1:]
    main(out_dir, ftype)
