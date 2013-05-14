import datetime as dt


def year_month(x):
    ym = (int(str(x)[:4]), int(str(x)[-2:]))
    return dt.datetime(ym[0], ym[1], 1)


def year(x):
    y = int(str(x)[:4])
    return dt.datetime(y, 01, 1)


def year_month_eurostat(x):
    year, month = (int(s) for s in x.split('M'))
    return dt.datetime(year, month, 1)


def year_month_bis(x):
    d = {'Jan': 1,
         'Feb': 2,
         'Mar': 3,
         'Apr': 4,
         'May': 5,
         'Jun': 6,
         'Jul': 7,
         'Aug': 8,
         'Sep': 9,
         'Oct': 10,
         'Nov': 11,
         'Dec': 12}
    parts = x.split('.')
    month = d[parts[0]]
    if parts[1].startswith(('0', '1')):
        year = int('20' + parts[1])
    else:
        year = int('19' + parts[1])
    return dt.datetime(year, month, 1)


def year_quarter_bis(x):
    d = {'Q1': 1,
         'Q2': 4,
         'Q3': 7,
         'Q4': 10}
    parts = x.split('-')
    year, month = int(parts[0]), d[parts[1]]
    return dt.datetime(year, month, 1)


def coibon_greenbook(x):
    if isinstance(x, float):
        x = str(x).split('.')[0]
        year = int('19' + x[-2:])
        day = int(x[-4:-2])
        month = int(x[:len(x) - 4])
        return dt.datetime(year, month, day)
    else:
        # Second part
        return x
