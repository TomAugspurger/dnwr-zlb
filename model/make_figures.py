#/Users/tom/python2.7/bin/
"""
Make all the plots used in the paper.

May have superficial differences due to .matplotlibrcs.
"""

from astroML.plotting import hist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_run as ar
from helpers import load_params, sample_path

def get_df(pi, lambda_, ws, params):
    """
    Give a pi, lambda pair, get the dataframe of wage choices
    sample from the steady state.
    """
    pths, shocks = sample_path(ws, params, lambda_=lambda_, w0=.9, nseries=1000,
                               nperiods=100)
    df = pd.DataFrame(pths)
    return df


def get_index(df):
    """
    A bit of a wasteful, hackish way to get an index to use for the various
    lags.
    """
    cts, idx, ax = hist(df.diff(1).loc[30], bins='scott', alpha=.35,
                        width=.0005)
    return idx

def plot_wage_change_dist(df, pi, lambda_, nperiods=4, log=True, figkwargs=None,
                          axkwargs=None):
    """
    Make and save the figure for the distribution of wage changes.

    figkwargs is a dict passed to the fig constructor.
    axkwargs is a dict passed to the axes constructor.
    """
    idx = get_index(df)
    SOME_SS = 30  # just some period in the steady state.
    diffs = range(1, nperiods + 1)

    if log:
        df = np.log(df)
        strlog = ', (log scale),'  # see title formatting
    else:
        strlog = ''

    t = pd.concat([df.diff(x).iloc[SOME_SS] for x in diffs],
                  axis=1, keys=diffs)
    _figkwargs = {'figsize':(13, 8)}  # Leading _ is internal.
    if figkwargs is not None:
        figkwargs_.update(figkwargs)

    _axkwargs = {}
    if axkwargs is not None:
        _axkwargs.update(axkwargs)


    fig, ax = plt.subplots(**_figkwargs)
    cts, idx, other = hist(t.values, histtype='bar', bins=idx,
                           label=['lag={}'.format(i) for i in diffs],
                           ax=ax, normed=True, **_axkwargs)

    ax.set_title('Across Periods{0} $\pi={1}$, $\lambda={2}$'.format(
        strlog, pi, lambda_))
    ax.legend()
    return fig, ax


def savefig_(fig, pi, lambda_):
    outname = './figures/pi{}lambda{}.pdf'.format(str(pi).replace('.', 'x'),
                                                  str(lambda_).replace('.', 'x'))
    fig.savefig(outname, format='pdf')
    plt.close()

def main():
    params = load_params()
    all_files = ar.get_all_files(params)
    wses = ar.read_output(all_files, kind='ws')
    keys = wses.keys()
    pis, lambdas = zip(*keys)  # FTW
    pis_u, lambdas_u = sorted(set(pis)), sorted(set(lambdas))  # unique

    for pi, lambda_ in keys:
        df = get_df(pi, lambda_, wses[pi, lambda_], params)
        fig, ax = plot_wage_change_dist(df, pi, lambda_)
        savefig_(fig, pi, lambda_)
        print('Saved {}, {}'.format(pi, lambda_))

if __name__ == '__main__':
    main()


