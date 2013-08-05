from collections import defaultdict
import pickle

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


if __name__ == '__main__':
    params = load_params()
    res_path = params['results_path'][0]
    try:
        res_dir = Path(res_path)
    except TypeError:
        res_dir = Path(str(res_path))

    binned = bin_results(res_dir)
    print(binned)
