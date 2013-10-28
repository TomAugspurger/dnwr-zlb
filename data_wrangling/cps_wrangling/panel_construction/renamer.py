"""
Some files came like dec00pub.zip, we would like cpsb0012.
"""

import gzip
import os
import shutil
import subprocess

import pathlib


def renamer(dir_):
    m_to_yr = {"jan": "01",
               "feb": "02",
               "mar": "03",
               "apr": "04",
               "may": "05",
               "jun": "06",
               "jul": "07",
               "aug": "08",
               "sep": "09",
               "oct": "10",
               "nov": "11",
               "dec": "12"}
    YEAR = slice(3, 5)
    for pth in dir_:
        f = pth.parts[-1]
        ext = '.' + f.split('.')[-1]
        if f.startswith(tuple(m_to_yr.keys())):
            print(pth)
            pth.rename(str(dir_) + '/' + 'cpsb' + f[YEAR] + m_to_yr[f[:3]] +
                       ext)


def full_year(dir_):
    all_files = iter(dir_)
    files = (x for x in all_files if x.parts[-1].startswith('cpsb'))
    for f in files:
        strf = f.parts[-1]
        year = int(strf[5:7])
        month = strf[6:8]
        if year < 50:  # TODO: will break in year 2050 :(
            year += 1900
        else:
            year += 2000
        year = str(year)
        name = os.path.join(str(dir_), year + '_' + month + '.gz')
        shutil.move(f, name)
        print("Moved {}".format(name))


def make_gzip(dir_):
    strdir = str(dir_)
    contents = os.listdir(strdir)
    need_recompressed = (os.path.join(strdir, x) for x in contents
                         if x.endswith(('.zip', 'Z')))

    for fname in need_recompressed:
        contents = set(os.listdir(strdir))
        out_name = fname.split('.')[0] + '.gz'
        subprocess.call(["7z", "x", fname])
        new_contents = set(os.listdir(strdir))
        new_files = new_contents - contents

        assert len(new_files) == 1
        new_files = list(new_files)[0]

        f_in = open(new_files, 'r')
        f_out = gzip.open(out_name, 'wb')
        f_out.writelines(f_in)
        os.remove(fname)
        os.remove(new_files)
        print("Compressed {}".format(fname))

if __name__ == '__main__':
    import json
    with open('settings.txt', 'r') as f:
        settings = json.load(f)

    dir_ = pathlib.PosixPath(str(settings['raw_monthly_path']))
    os.chdir(str(dir_))
    renamer(dir_)
    make_gzip(dir_)
    full_year(dir_)
