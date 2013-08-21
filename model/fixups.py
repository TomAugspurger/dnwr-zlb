"""
Cleaning up the inevible bugs without rerunning uncesscessary parts.

Try to keep a log of runs here.

Date       | Method                 | Reason
#-----------------------------------|-----------------------------------------
2013-08-11 | fixup_bad_gps          | Solo valid grid points
2013-08-15 | fixup_bad_outputs      | Messed up integration
2013-08-16 | fixup_bad_gps (part 2) | Freshman's dream on gp(X * (1 + pi))
2013-08-16 | fixup_bad_out (part 2) | Follup to last error
2013-08-19 | fix bad gps > out      | undoing previous dumbness
2013-08-21 | refactor gp to ecdf    | major refactor
#-----------------------------------------------------------------------------
"""
import datetime
import cPickle
import shutil

import numpy as np

import analyze_run as ar
from ecdf import ecdf
from gen_interp import Interp
from helpers import load_params, ss_wage_flexible, sample_path
from value_function import get_rigid_output


def wage_dist_ecdf_refactor():
    """
    This implements the refactor of gp's to ecdfs.

    Also takes care of the output.
    """
    with open('results/fixup_notice.txt', 'a') as f:
        t = str(datetime.datetime.now())
        f.write("FIXED gps AT {}\n".format(t))

    params = load_params()
    params['results_path/'] = 'results/', 'a'
    all_files = ar.get_all_files(params)
    gps = ar.read_output(all_files, kind='gp')
    wses = ar.read_output(all_files, kind='ws')
    z_grid = params['z_grid'][0]
    flex_ws = Interp(z_grid, ss_wage_flexible(params, shock=z_grid))

    for key in gps.iterkeys():
        piname, lambda_ = [str(x).replace('.', '') for x in key]
        out_name = 'results/gp_' + piname + '_' + lambda_ + '.pkl'
        shutil.copy2(out_name, 'results/replaced_results/')
        ws = wses[key]
        params['pi'] = key[0], 'you'
        params['lambda_'] = key[1], 'idiot'
        new_g, shocks = get_new_g(ws, params)

        with open(out_name, 'w') as f:
            cPickle.dump(new_g, f)

        print("Fixed wage distribution for {}.".format(key))

        new_rigid_out = get_rigid_output(ws, params, flex_ws, new_g, shocks)

        out_name = 'results/rigid_output_' + piname + '_' + lambda_ + '_.txt'
        with open(out_name, 'w') as f:
            f.write(str(new_rigid_out))

        with open('results/fixup_notice.txt', 'a') as f:
            f.write("Fixed {}\n".format(key))


def get_new_g(ws, params):
    np.random.seed(42)
    pths, shks = sample_path(ws, params, nseries=1000, nperiods=30)
    pth, shocks = pths[28], shks[28]  # a period in steady state
    g = ecdf(np.sort(pth))
    shocks = np.sort(shocks)
    return g, shocks


if __name__ == '__main__':
    wage_dist_ecdf_refactor()
