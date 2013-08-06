from collections import defaultdict
import os
import pickle
import re

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


def read_output(fnames):
    """
    Construct a dict of {pi: output} given a list of files.

    Example
    all_files = os.listdir('results/')
    res = read_output(all_files)
    """
    # Trouble with where they run it from.
    if not os.path.exists(fnames[1]):
        prepend = 'results/'
    else:
        prepend = ''
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

if __name__ == '__main__':
    params = load_params()
    res_path = params['results_path'][0]
    try:
        res_dir = Path(res_path)
    except TypeError:
        res_dir = Path(str(res_path))

    binned = bin_results(res_dir)
    print(binned)
