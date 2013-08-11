from __future__ import division

from datetime import datetime
import itertools
import json
import pickle

from helpers import load_params, ss_wage_flexible
from joblib import Parallel, delayed
import numpy as np

from gen_interp import Interp
from value_function import get_rigid_output, g_p, iter_bellman


def iter_bellman_wrapper(hyperparams):
    """
    Call this from a joblib Parallel like.

    Parameters
    ----------

    hyperparams: inflation rate, rigidity tuple.

    Returns
    -------
    None:  Does have side effects.
    """
    pi, lambda_ = hyperparams
    np.random.seed(42)
    params = load_params()
    params['pi'] = pi, 'inflation target'
    params['lambda_'] = lambda_, 'rigidity'
    w_grid = params['w_grid'][0]
    z_grid = params['z_grid'][0]

    write_metadeta(params)

    v = Interp(w_grid, -w_grid + 29, kind='cubic')
    Tv, ws, rest = iter_bellman(v, tol=0.005, strict=False, log=False,
                                params=params, pi=pi)
    res_dict = {'Tv': Tv, 'ws': ws, 'rest': rest}
    #-------------------------------------------------------------------------
    flex_ws = Interp(z_grid, ss_wage_flexible(params, shock=z_grid))
    #-------------------------------------------------------------------------
    gp = get_wage_distribution(ws, params)
    res_dict['gp'] = gp
    #-------------------------------------------------------------------------
    rigid_out = get_rigid_output(ws, params, flex_ws, gp)
    res_dict['rigid_out'] = rigid_out
    #-------------------------------------------------------------------------
    write_results(res_dict, pi, lambda_)
    pass


def get_wage_distribution(ws, params):
    w_grid = params['w_grid'][0]
    # fine_grid = params['fine_grid'][0]
    w_max = w_grid[-1]
    g0 = Interp(w_grid, w_grid/w_max, kind='pchip')
    gp = g_p(g0, ws, params)
    return gp


def write_results(res_dict, pi, lambda_):
    """
    Handle the file writing of iter_bellman.

    Parameters
    ----------
    """
    piname = str(pi).replace('.', '')
    lname = str(lambda_).replace('.', '')
    out_name = '_'.join([piname, lname])
    with open('results/vf_' + out_name + '.pkl', 'w') as f:
        pickle.dump(res_dict['Tv'], f)
    with open('results/ws_' + out_name + '.pkl', 'w') as f:
        pickle.dump(res_dict['ws'], f)
    with open('results/gp_' + out_name + '.pkl', 'w') as f:
        pickle.dump(res_dict['gp'], f)
    with open('results/rigid_output_' + out_name + '_.txt', 'w') as f:
        f.write(str(res_dict['rigid_out']))
    res_dict['rest'].to_hdf('results/results_' + out_name + '.h5',
                            'pi_' + out_name)
    print('Added results for {}'.format(out_name))


def unique_param_generator(params):
    """
    This is where we'll use argparse to generalize.

    When you generalize add some kind of id for what is special about
    that run.  Use that for filenaming etc.  E.g. if going over
    lambda x pi space, set params['id'] = (lambda, pi)
    """
    pi_low = params['pi_low'][0]
    pi_high = params['pi_high'][0]
    pi_n = params['pi_n'][0]
    pi_grid = np.linspace(pi_low, pi_high, pi_n)

    lambda_l = params['lambda_l'][0]
    lambda_u = params['lambda_u'][0]
    lambda_n = params['lambda_n'][0]
    lambda_grid = np.linspace(lambda_l, lambda_u, lambda_n)

    hyper_space = itertools.product(pi_grid, lambda_grid)
    for pi, lambda_ in hyper_space:
        params['pi'] = pi, 'Target inflation rate.'
        params['lambda_'] = lambda_, 'rigidity'
        yield params


def write_metadeta(params, outname='metadata.json'):
    time = str(datetime.now())
    params['time'] = time, 'start of run'

    # arrays/dists aren't JSON serializable.  Can be reconstruncted anyway
    all_keys = {x for x in params.keys()}
    nonserial = {'z_grid', 'w_grid', 'z_grid_fine', 'w_grid_fine',
                 'full_ln_dist'}

    writeable = all_keys - nonserial
    out_dict = {k: v for k, v in params.iteritems() if k in writeable}
    with open(outname, 'w') as f:
        json.dump(out_dict, f)


if __name__ == '__main__':
    import sys
    params_path = sys.argv[1]
    # keep load_params outside so that each fork has the same random seed.
    np.random.seed(42)
    params = load_params(params_path)
    pi_low = params['pi_low'][0]
    pi_high = params['pi_high'][0]
    pi_n = params['pi_n'][0]
    pi_grid = np.linspace(pi_low, pi_high, pi_n)
    # Parallel(n_jobs=-1)(delayed(iter_bellman_wrapper)(unique_params)
                        # for unique_params in unique_param_generator(params))
    lambda_l = params['lambda_l'][0]
    lambda_u = params['lambda_u'][0]
    lambda_n = params['lambda_n'][0]
    lambda_grid = np.linspace(lambda_l, lambda_u, lambda_n)
    hyperspace = itertools.product(pi_grid, lambda_grid)
    Parallel(n_jobs=-1)(delayed(iter_bellman_wrapper)(tup)
                        for tup in hyperspace)
