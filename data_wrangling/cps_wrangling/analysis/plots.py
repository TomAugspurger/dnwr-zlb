import itertools as it
import json

import matplotlib.pyplot as plt
import pandas as pd


def iter_wrapper(func, iterable, *args, **kwargs):
    fig, ax = plt.subplots()
    colors = it.cycle(plt.rcParams['axes.color_cycle'])
    for x in iterable:
        color = next(colors)
        yield func(x, color=color, axis=ax, **kwargs)


def earnings_scatter(df, **kwargs):
    """
    Takes a df from earn_store.
    """
    earn = df[['PRERNWA1', 'PRERNWA2']]
    ax = kwargs.pop('axis', None)
    if ax is None:
        fig, ax = plt.subplots()
    return ax.scatter(earn['PRERNWA1'], earn['PRERNWA2'],
                      marker='.', s=12, alpha=.4, **kwargs)


def earnings_diff_kde(df, **kwargs):
    earn = df[['PRERNWA1', 'PRERNWA2']].dropna()
    ax = kwargs.pop('axis', None)
    if ax is None:
        fig, ax = plt.subplots()
    return (earn.PRERNWA2 - earn.PRERNWA1).plot(kind='kde', **kwargs)


def get_earn_store():
    with open('../panel_construction/settings.txt', 'r') as f:
        settings = json.load(f)

    earn_store = pd.HDFStore(settings["earn_store_path"])
    return earn_store


def main():
    earn_store = get_earn_store()



if __name__ == '__main__':
    main()
