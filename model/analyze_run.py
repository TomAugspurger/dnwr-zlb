from collections import defaultdict
import itertools as it
import os
import pickle
import re

import pandas as pd
from pathlib import Path

from helpers import load_params


def bin_results(res_dir):
    """
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
    reg = re.compile(r"[tst_]*rigid_output_(\d*).*")
    pi_out_dict = {}
    for fname in fnames:
        # Change the first zero to a decimal point that was
        # removed during writing.
        try:
            pi = float(reg.match(fname).groups()[0].replace('0', '.', 1))
        except AttributeError:
            continue
        with open(prepend + fname, 'rb') as f:
            pi_out_dict[pi] = float(f.read())
    return pi_out_dict


def _read_pickleable(fnames, kind, prepend=''):
    reg = re.compile(r"[tst_]*" + kind + "_(\d*).*")
    pi_dict = {}
    for fname in fnames:
        # Change the first zero to a decimal point that was
        # removed during writing.
        try:
            pi = float(reg.match(fname).groups()[0].replace('0', '.', 1))
        except AttributeError:
            continue
        with open(prepend + fname, 'rb') as f:
            pi_dict[pi] = pickle.load(f)
    return pi_dict


def _read_results(fnames, prepend=''):
    return pd.HDFStore(prepend + 'grouped_results.h5')


def group_results(fnames, prepend=''):
    reg = re.compile(r"[tst_]*results_(\d*).*")

    def filt(x):
        if x[1] is None:
            return False
        else:
            return True

    matches = it.izip(fnames, map(reg.match, fnames))
    filtered = it.ifilter(filt, matches)
    dic = {x[1].groups()[0]: x[0] for x in filtered}
    store = pd.HDFStore(prepend + 'grouped_results.h5')

    for key, f in dic.iteritems():
        pan = pd.read_hdf(prepend + f, 'pi_' + key)
        store.append('pi_' + key, pan)
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
            group_results(fnames, 'prepend')
        return _read_results(fnames, prepend)
    else:
        raise ValueError("Kind must be one of 'rigid_output'",
                         "'vf', 'ws', 'gp', or 'results'.",
                         " Got {} instead.".format(kind))
if __name__ == '__main__':
    params = load_params()
    res_path = params['results_path'][0]
    try:
        res_dir = Path(res_path)
    except TypeError:
        res_dir = Path(str(res_path))

    binned = bin_results(res_dir)
    print(binned)
