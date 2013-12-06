import pandas as pd
from lxml import html

def add_rec_bars(ax, df=None, dates=None, **kwargs):
    # p = html.parse('http://research.stlouisfed.org/fred2/help-faq/#graph_recessions')
    # r = p.getroot()
    # d = r.body.get_element_by_id('content').get_element_by_id(
    #     'content-container').get_element_by_id('content-2columns-main')

    if dates is None:
        dates = pd.read_csv('/Users/tom/bin/rec_dates.csv',
                            parse_dates=['Peak', 'Trough'])
    y1, y2 = ax.get_ylim()
    if df:
        f = df.index.freq
    else:
        f = 'MS'
    # build the ranges:
    ranges = [pd.date_range(start=x[1]['Peak'], end=x[1]['Trough'],
              freq=f) for x in dates.iterrows()]

    alpha = kwargs.pop('alpha', .25)
    color = kwargs.pop('color', 'k')

    for span in ranges:
        ax.fill_between(span, y1=y1, y2=y2, alpha=alpha, color=color, **kwargs)
    ax.set_ylim(y1, y2)
    return ax
