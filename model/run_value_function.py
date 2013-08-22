#!/Users/tom/python2.7/bin/python

from __future__ import division

from datetime import datetime
import itertools
import json
import os
import pickle

from helpers import load_params, ss_wage_flexible, sample_path
from joblib import Parallel, delayed
import numpy as np

from ecdf import ecdf
from gen_interp import Interp
from value_function import get_rigid_output, iter_bellman


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
    piname = str(pi).replace('.', '')
    lname = str(lambda_).replace('.', '')
    out_name = '_'.join([piname, lname])
    params = load_params()
    params['pi'] = pi, 'inflation target'
    params['lambda_'] = lambda_, 'rigidity'

    # check for pre-computed values.
    params['call_dir'] = os.getcwd(), 'Path from which the script was called.'
    if os.path.exists(
            params['call_dir'][0] + '/results/' + 'vf_' + out_name + '.pkl'):
        print("Skipping pi:{}, lambda:{}.".format(pi, lambda_))
        return None

    res_dict = run_one(params)
    output_e = 1
    output_tol = .005
    while output_e > output_tol:
        out_now = res_dict['rigid_out']
        res_dict = run_one(params, res_dict)
        out_next = res_dict['rigid_out']
        output_e = np.abs(out_now - out_next)
        print("The change in output was {}.".format(output_e))
    #-------------------------------------------------------------------------
    write_results(res_dict, pi, lambda_)
    pass


def run_one(params, res_dict=None):
    """
    Once you have the parameters, this function completes one loop to get
    a dictionary of results.

    For the first loop leave res_dict as None.
    """
    pi = params['pi'][0]
    # lambda_ = params['lambda_'][0]

    np.random.seed(42)
    w_grid = params['w_grid'][0]
    z_grid = params['z_grid'][0]

    if res_dict:
        v = res_dict['Tv']
        out = res_dict['rigid_out']
    else:
        v = Interp(w_grid, -w_grid + 29, kind='linear')
        out = 0.85049063822172699  # ss output w/ flexible wages
    # Get close with linear first.  Then do a few cubic to finish up
    Tv, ws, rest = iter_bellman(v, tol=0.005, strict=False, log=False,
                                params=params, pi=pi, aggL=out, kind='linear')
    Tvc = Interp(Tv.X, Tv.Y, kind='cubic')
    Tv, ws, rest = iter_bellman(Tvc, tol=0.005, strict=False, log=False,
                                params=params, pi=pi)
    res_dict = {'Tv': Tv, 'ws': ws, 'rest': rest}
    flex_ws = Interp(z_grid, ss_wage_flexible(params, shock=z_grid))
    #-------------------------------------------------------------------------
    pths, shks = sample_path(ws, params, nseries=1000, nperiods=30)
    pth, shocks = pths[28], shks[28]  # a period in steady state
    g = ecdf(np.sort(pth))
    shocks = np.sort(shocks)
    #-------------------------------------------------------------------------
    rigid_out = get_rigid_output(ws, params, flex_ws, g, shocks)
    res_dict['gp'] = g
    res_dict['rigid_out'] = rigid_out
    return res_dict


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
    with open('results/' + outname, 'w') as f:
        json.dump(out_dict, f)


def get_g(pi, lambda_, period=28):
    """
    Helper function to get to a wage distribution.

    Warning: Will not touch the params in your global state.
    If you go on the to more things make sure to adjust those params.
    """
    import analyze_run as ar
    params = load_params()
    params['pi'] = pi, 'a'
    params['lambda_'] = lambda_, 'b'

    all_files = ar.get_all_files(params)
    wses = ar.read_output(all_files, kind='ws')
    ws = wses[(pi, lambda_)]
    pths, shks = sample_path(ws, params, nseries=1000, nperiods=30, seed=42)

    pth, shocks = pths[28], shks[28]
    shocks = np.sort(shocks)
    g = ecdf(np.sort(pth))
    return g, shocks

# When is rigid output higher / lower than rigid
# all comes to p1 vs. (p2 + p3)

if __name__ == '__main__':
    import sys
    params_path = sys.argv[1]
    # keep load_params outside so that each fork has the same random seed.
    np.random.seed(42)
    params = load_params(params_path)

    write_metadeta(params)

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
