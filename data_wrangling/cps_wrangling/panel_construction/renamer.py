"""
Some files came like dec00pub.zip, we would like cpsb0012.
"""
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

if __name__ == '__main__':
    import sys
    dir_ = pathlib.Path(sys.argv[1])
    renamer(dir_)
