#!/Users/tom/python2.7/bin/python

from __future__ import division

from collections import defaultdict
from datetime import datetime
import itertools
import json
import os
import pickle

from helpers import load_params, ss_wage_flexible, sample_path, ss_output_flexible
from joblib import Parallel, delayed
import numpy as np

from ecdf import ecdf
from gen_interp import Interp
from value_function import get_rigid_output, iter_bellman


class BellmanRunner(object):

    def __init__(self, hyperparams):

        self.pi, self.lambda_ = hyperparams
        self.piname = str(self.pi).replace('.', '')
        self.lname = str(self.lambda_).replace('.', '')
        self.out_name = '_'.join([self.piname, self.lname])
        params = load_params()
        params['pi'] = self.pi, 'inflation target'
        params['lambda_'] = self.lambda_, 'rigidity'

        params['call_dir'] = os.getcwd(), 'Path from which the script was called.'
        self.params = params

        self.res_by_run = []
        self.res_by_cat = defaultdict(list)

    def __call__(self):
        res_dict = self.generate_res_input()
        res_dict = run_one(self.params, res_dict=res_dict)
        return res_dict

    def generate_res_input(self):
        """
        Try to be smart about using past runs.

        If alternating use that.  If monotonic use the last one. Could
        *generalitze* filtering down to alternating and take the mean of that.
        """
        if len(self.res_by_run) == 0:
            res_dict = None
        elif len(self.res_by_run) == 1:
            res_dict = self.res_by_run[0]
        else:
            outs = np.array(self.res_by_cat['rigid_out'])
            Tvs = self.res_by_cat['Tv']
            X = Tvs[0].X
            if np.all(np.abs(np.diff(np.sign(np.diff(outs))))):
                # alternating
                out = np.mean(outs)
                Y = np.mean([v.Y for v in Tvs])
                v = Interp(X, Y, kind='linear')
            else:
                out = outs[-1]
                v = Tvs[-1]
            res_dict = {'Tv': v, 'rigid_out': out}

        return res_dict

    def from_dict(self):
        """
        Good for analysis.
        """
        raise NotImplemented

    def update(self, res_dict):
        """
        Add the results to both instance dictionaries.

        Returns None.
        """
        self.res_by_run.append(res_dict)
        self.res_by_cat['gp'].append(res_dict['gp'])
        self.res_by_cat['vf'].append(res_dict['Tv'])
        self.res_by_cat['ws'].append(res_dict['ws'])
        self.res_by_cat['rigid_out'].append(res_dict['rigid_out'])
        self.last = res_dict

    def iter_over_output(self, tol=.05):
        e = 1
        out_prior = ss_output_flexible(self.params)
        while e > tol:
            res_dict = self()
            self.update(res_dict)
            out = res_dict['rigid_out']
            e = np.abs(out_prior - out)
            print('The new error is {}'.format(e))
            out_prior = out
        else:
            self.terminating_e = e

    def write_results(self):
        """
        Writes last one to results/
        and others to results/intermediate/
        """
        res_dict = self.last
        write_results(res_dict, self.pi, self.lambda_)
        for i, d in enumerate(self.res_by_run):
            write_results(d, self.pi, self.lambda_, intermediate=True, i=i)


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

    kls = BellmanRunner(hyperparams)
    kls.iter_over_output()
    kls.write_results()


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
        print("Reusing res_dict.")
        v = res_dict['Tv']
        out = res_dict['rigid_out']
    else:
        v = Interp(w_grid, -w_grid + 27.3, kind='linear')
        out = ss_output_flexible(params)  # ss output w/ flexible wages
    # Get close with linear first.  Then do a few cubic to finish up
    Tv, ws, rest = iter_bellman(v, tol=0.005, strict=False, log=False,
                                params=params, pi=pi, aggL=out, kind='linear')
    Tvc = Interp(Tv.X, Tv.Y, kind='cubic')
    Tv, ws, rest = iter_bellman(Tvc, tol=0.005, strict=False, log=False,
                                params=params, pi=pi, aggL=out)
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


def write_results(res_dict, pi, lambda_, intermediate=False, i=''):
    """
    Handle the file writing of iter_bellman.

    Parameters
    ----------
    """
    piname = str(pi).replace('.', '')
    lname = str(lambda_).replace('.', '')
    out_name = '_'.join([piname, lname])

    if intermediate:
        mid = 'intermediate/'
        i = str(i)
    else:
        mid = ''
        i = ''

    with open('results/' + mid + 'vf_' + out_name + i + '.pkl', 'w') as f:
        pickle.dump(res_dict['Tv'], f)
    with open('results/' + mid + 'ws_' + out_name + i + '.pkl', 'w') as f:
        pickle.dump(res_dict['ws'], f)
    with open('results/' + mid + 'gp_' + out_name + i + '.pkl', 'w') as f:
        pickle.dump(res_dict['gp'], f)
    with open('results/' + mid + 'rigid_output_' + out_name + i + '_.txt', 'w') as f:
        f.write(str(res_dict['rigid_out']))
    res_dict['rest'].to_hdf('results/' + mid + 'results_' + out_name + i + '.h5',
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


def reloader(pth, out_name):
    """
    If you have a run that you want to iterate again,
    but you want to start from the existing vf, ws, etc.
    """
    vf_path = os.path.join(pth, 'vf_' + out_name + '.pkl')
    with open(vf_path, 'r') as f:
        Tv = (pickle.load(f))

    output_path = os.path.join(pth, 'rigid_output_' + out_name + '_.txt')
    with open(output_path, 'r') as f:
        rigid_out = float(f.read())

    res_dict = {'Tv': Tv, 'rigid_out': rigid_out}

    return res_dict


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

    os.makedirs('./results/intermediate')
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
