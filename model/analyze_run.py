#/Users/tom/python2.7/bin/python

from collections import defaultdict, Iterable
import itertools as it
import json
import os
import pickle
import re

from astroML.plotting import hist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import interpolate

from helpers import load_params, sample_path
from value_function import u_

def bin_results(res_dir):
    """
    Essentially deprecatied.

    Take a Path object and return a dict based on type of result.

    Parameters
    ----------

    res_dir: pathlib.Path iterable object

    Returns
    -------

    binned_results : dict of type -> filename
    """
    binned_results = defaultdict(list)
    groups = ['results', 'vf', 'ws', 'gp', 'rigid_output']
    for child in res_dir:
        if child.match('.DS_Store'):
            continue
        for group in groups:
            if child.name.startswith(group):
                binned_results[group].append(str(child))
    return binned_results


def load_pickleable(pth):
    """
    Use in a comprehension for ws, vf, gp
        pth is a str.
    """
    with open(pth, 'r') as f:
        res = (pickle.load(f))
    return res


def load_single(pth):
    """
    Use for things like rigid_output, i.e. just single lines.
    """
    with open('pth', 'r') as f:
        res = float(f.read())
    return res


def _read_rigid_output(fnames, prepend=''):
    """
    Construct a dict of {pi: output} given a list of files.

    Example
    all_files = os.listdir('results/')
    res = read_output(all_files)
    """
    # Trouble with where they run it from.
    reg = re.compile(r"[tst_]*rigid_output_(\d*)_(\d*).*")
    pi_out_dict = {}
    for fname in fnames:
        # Change the first zero to a decimal point that was
        # removed during writing.
        try:
            match = reg.match(fname)
            pi = float(match.groups()[0].replace('0', '.', 1))
            lambda_ = float(match.groups()[1].replace('0', '.', 1))
        except AttributeError:
            continue
        with open(prepend + fname, 'r') as f:
            try:
                pi_out_dict[(pi, lambda_)] = float(f.read())
            except ValueError:
                f.seek(0)
                val = json.load(f)
                pi_out_dict[(pi, lambda_)] = float(val[0])
    return pi_out_dict


def _read_pickleable(fnames, kind, prepend=''):
    reg = re.compile(r"[tst_]*" + kind + "_(\d*)_(\d*).*")
    pi_dict = {}
    for fname in fnames:
        # Change the first zero to a decimal point that was
        # removed during writing.
        try:
            match = reg.match(fname)
            pi = float(match.groups()[0].replace('0', '.', 1))
            lambda_ = float(match.groups()[1].replace('0', '.', 1))
        except AttributeError:
            continue
        with open(prepend + fname, 'rb') as f:
            pi_dict[(pi, lambda_)] = pickle.load(f)
    return pi_dict


def _read_results(fnames, prepend=''):
    return pd.HDFStore(prepend + 'grouped_results.h5')


def group_results(fnames, prepend=''):
    reg = re.compile(r"[tst_]*results_(\d*)_(\d*).*")

    def filt(x):
        if x[1] is None:
            return False
        else:
            return True

    matches = it.izip(fnames, map(reg.match, fnames))
    filtered = it.ifilter(filt, matches)
    dic = {x[1].groups(): x[0] for x in filtered}
    store = pd.HDFStore(prepend + 'grouped_results.h5')

    for key, f in dic.iteritems():
        storename = '_'.join(key)
        pan = pd.read_hdf(prepend + f, 'pi_' + storename)
        store.append('pi_' + '_lambda_'.join(key), pan)
    else:
        store.close()


def read_output(fnames, kind):
    """
    Public API for reading files of type kind.
    kind is one of:
        - rigid_output :: float for output
        - vf :: Interp for value function
        - gp :: Interp for wage dist
        - ws :: Interp for wage schedule
        - results :: HDFStore of panels
    """
    if not os.path.exists(fnames[1]):
        prepend = 'results/'
    else:
        prepend = ''

    if kind == 'rigid_output':
        return _read_rigid_output(fnames, prepend)
    elif kind in ('vf', 'gp', 'ws'):
        return _read_pickleable(fnames, kind=kind, prepend=prepend)
    elif kind == 'results':
        if not os.path.exists('results/grouped_results.h5'):
            group_results(fnames, prepend)
        return _read_results(fnames, prepend)
    else:
        raise ValueError("Kind must be one of 'rigid_output'",
                         "'vf', 'ws', 'gp', or 'results'.",
                         " Got {} instead.".format(kind))


def get_all_files(params=None):
    """Get the files from the results path"""
    if params is None:
        params = load_params()

    pth = params['results_path'][0]
    all_files = os.listdir(pth)
    return all_files

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------


def filter_(dic, pi=None, lambda_=None):
    """
    Given a dict of results like {(pi, lambda): val}, filter
    down to just those with pi or lambda_.

    Parameters
    ----------

    dic: dictionary of results
    pi: float/array of floats in dic keys
    lambda_: float/array of floats in dic keys

    Returns
    -------

    filtered_dict
    """
    if pi is None and lambda_:
        if not isinstance(lambda_, Iterable):
            lambda_ = (lambda_,)
        cond = set(it.product((x[0] for x in dic.keys()), lambda_))
    elif pi and lambda_ is None:
        if not isinstance(pi, Iterable):
            pi = (pi,)
        cond = set(it.product(pi, (x[1] for x in dic.keys())))
    elif pi and lambda_:
        if not isinstance(pi, Iterable):
            pi = (pi,)
        if not isinstance(lambda_, Iterable):
            lambda_ = (lambda_,)
        cond = set(it.product(pi, lambda_))
    else:
        raise ValueError("At least one must be not None.")

    return {key: val for key, val in dic.iteritems() if key in cond}


def flatten(dic):
    """
    Take a dict and make it a 2-d array of (pi, lambda, value)
    """
    return np.array([(key[0], key[1], val) for key, val in dic.iteritems()])


def interp_output(out_dict, kind='cubic'):
    """
    2-d interpolation over (pi x lambda) -> output
    """
    out_ar = flatten(out_dict)
    return interpolate.interp2d(out_ar[:, 0], out_ar[:, 1], out_ar[:, 2],
                                kind=kind)


def alt_interp(out_dict):
    """
    Just a dump.  Haven't tested.  Especially don't trust the grids and
    having the axis correct.  Don't trust anything.
    """
    out_ar = flatten(out_dict)
    pis = sorted(list({x[0] for x in out_dict}))
    lambdas_ = sorted(list({x[1] for x in out_dict}))
    grid_pi = np.linspace(pis[0], pis[-1], 500)
    grid_l = np.linspace(lambdas_[0], lambdas_[-1], 500)
    grid1, grid2 = np.meshgrid(grid_pi, grid_l)
    z_l = interpolate.griddata(out_ar[:, 0:2], out_ar[:, 2], (grid1, grid2), method='linear')
    z_c = interpolate.griddata(out_ar[:, 0:2], out_ar[:, 2], (grid1, grid2), method='cubic')
    z_n = interpolate.griddata(out_ar[:, 0:2], out_ar[:, 2], (grid1, grid2), method='nearest')
    return z_l, z_c, z_n


def get_outs_df(out_dict):
    """
    Turn a dictionary of (pi x lambda : output) into a dataframe.
    """
    idx = pd.MultiIndex.from_tuples(out_dict.keys(), names=['pi', 'lambda_'])
    df = pd.DataFrame(out_dict.values(), columns=['output'], index=idx)
    df = df.sort_index()
    return df


def make_panel(wses, params, pairs=None, log=False, nseries=100, nperiods=50,
               seed=42):
    """
    Panel of wage choices where

        items : (pi, lambda)
        major : periods
        minor : series

    seed is an int given to np.random.set_seed().
    """
    if pairs is None:
        pairs = wses.iterkeys()

    dfs = {}
    sks = {}
    for key in pairs:
        ws = wses[key]
        pths, shocks = sample_path(ws, params, w0=.9, lambda_=key[1],
                                   nseries=nseries, nperiods=nperiods,
                                   seed=seed)
        df = pd.DataFrame(pths)
        shocks = pd.DataFrame(shocks)
        if log:
            dfs[key] = np.log(df)
        else:
            dfs[key] = df
        sks[key] = shocks
    df_pan = pd.Panel(dfs)
    sk_pan = pd.Panel(sks)
    df_pan.minor_axis.name = 'person'
    sk_pan.minor_axis.name = 'person'
    df_pan.items.name = 'key'
    sk_pan.items.name = 'key'
    df_pan.major_axis.name = 'period'
    sk_pan.major_axis.name = 'period'
    return df_pan, sk_pan


def make_hist(pan, subpairs, ax=None, **kwargs):
    t = pd.concat([pan.xs(x, axis='items').diff().iloc[30] for x in subpairs],
                  axis=1)
    t.columns = map(str, subpairs)
    _, idx1, fig = hist(t[t.columns[0]], bins='scott', alpha=.35, width=.0005)
    plt.close()
    if ax is None:
        fig = plt.figure(**kwargs)
        ax = fig.add_subplot(111)
    return hist(t.values, bins=idx1, ax=ax, histtype='bar',
                label=t.columns.tolist())


def mean_with_ci(df, lower=.05, upper=.95):
    tdf = pd.DataFrame({'mean': df.mean(), 'lower': df.quantile(lower),
                        'upper': df.quantile(upper)})
    return tdf


def plot_with_ci(df, lower=.05, upper=.95, **kwargs):
    tdf = mean_with_ci(df, lower, upper)
    fig, ax = plt.subplots()
    ## Need to figure out who gets which kwargs.  Maybe use a get w/ defaults
    tdf['mean'].plot(ax=ax, **kwargs)
    ax.fill_between(tdf.index, tdf['lower'].values, tdf['upper'].values,
                    alpha=.4)
    return fig, ax


def tuple_constructor(dic):
    """
    Most of my results are of {(pi. lambda) : value} type.

    This helper unpacks the pi x lambda tupes, and returns the Dict as
    a Series or DataFrame with a MultiIndex.
    """
    keys = dic.keys()
    df = pd.DataFrame.from_dict(dic, orient='index')
    df.index = pd.MultiIndex.from_tuples(keys, names=['pi', 'lambda_'])
    return df


def plot_sub(df, pi=None, lambda_=None, **kwargs):
    """
    Use if you have series and want to plot just a subset, eg with
    pi = .05.  Haven't tested for DataFrames.
    """
    dic = df.to_dict()
    subdf = type(df)(filter_(dic, pi, lambda_))
    pis, lambdas_ = zip(*subdf.index)  # Python win


def get_utils(wpan, span, out_ser, by_lam=False):
    keys = wpan.items
    dfu = {}
    for key in keys:
        dfw = wpan[key]
        dfs = span[key]
        dfu[key] = u_(dfw, shock=dfs, aggL=out_ser.ix[key]).mean()

    dfu = tuple_constructor(dfu)
    if by_lam:
        dfu = dfu.swaplevel(0, 1)
    dfu = dfu.sort_index()
    return dfu


def up(hrs, shock, output, eta=2.5, gamma=.5):
    p1 = np.log(output)
    p2 = (gamma / (gamma + 1)) * shock * hrs ** ((gamma + 1) / gamma)
    return p1 - p2
