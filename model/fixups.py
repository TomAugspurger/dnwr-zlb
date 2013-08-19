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
#-----------------------------------------------------------------------------
"""
import datetime
import pickle
import shutil

import analyze_run as ar
from gen_interp import Interp
from helpers import load_params, ss_wage_flexible
from run_value_function import get_wage_distribution
from value_function import get_rigid_output


def fixup_bad_gps():
    """
    Got flat wage distributions due to solo valid grid points.

    See commit `e5ace15` and tag `simulation3`.
    """
    with open('results/fixup_notice.txt', 'a') as f:
        t = str(datetime.datetime.now())
        f.write("FIXED gps AT {}\n".format(t))

    params = load_params()
    params['results_path/'] = 'results/', 'a'
    all_files = ar.get_all_files(params)
    gps = ar.read_output(all_files, kind='gp')

    # def _get_bad_gps(gp, cutoff=.5):
    #     """Ones whose max - min is less than, say, .5"""
    #     if gps[gp].Y[-1] - gps[gp].Y[0] < cutoff:
    #         return True
    #     else:
    #         return False

    # bad_gps_keys = filter(_get_bad_gps, gps)
    # bad_gps = {key: gps[key] for key in bad_gps_keys}

    for key in gps:
        piname, lambda_ = [str(x).replace('.', '') for x in key]
        out_name = 'results/gp_' + piname + '_' + lambda_ + '.pkl'
        shutil.copy2(out_name, 'results/replaced_results/')
        _replace(key, out_name, all_files, params)
        print("Fixed {}".format(key))
        with open('results/fixup_notice.txt', 'a') as f:
            f.write("Fixed {}\n".format(key))


def _replace(key, out_name, all_files, params):
    wses = ar.read_output(all_files, kind='ws')
    ws = wses[key]
    gp = get_wage_distribution(ws, params)
    with open(out_name, 'w') as f:
        pickle.dump(gp, f)

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------


def fixup_bad_outputs():
    """
    I wasn't handling the integrals correctly.  Before I was just
    summing, but that ignored the probability distribution for each
    shock.

    That's fixed now (hopefully).  Fortunately, the calculation
    of all the equilibrium stuff (vf, ws) is independent of the output
    calculation.  So we can pick up from there.
    """
    with open('results/fixup_notice.txt', 'a') as f:
        t = str(datetime.datetime.now())
        f.write("FIXED outputs AT {}\n".format(t))

    params = load_params()
    params['results_path/'] = 'results/', 'a'
    all_files = ar.get_all_files(params)

    gps = ar.read_output(all_files, kind='gp')
    wses = ar.read_output(all_files, kind='ws')
    z_grid = params['z_grid'][0]
    flex = ss_wage_flexible(params, shock=z_grid)
    flex_ws = Interp(z_grid, flex)

    for key in wses:
        ws = wses[key]
        gp = gps[key]
        params['pi'] = key[0], 'a'
        params['lambda_'] = key[1], 'b'

        piname, lambda_ = [str(x).replace('.', '') for x in key]
        out_name = 'results/rigid_output_' + piname + '_' + lambda_ + '_.txt'
        shutil.copy2(out_name, 'results/replaced_results/')

        new_out = get_rigid_output(ws, params, flex_ws, gp)

        with open(out_name, 'w') as f:
            f.write(str(new_out))

        with open('results/fixup_notice.txt', 'a') as f:
            f.write('Fixed output for {}.\n'.format(key))
            print("Fixed {}".format(key))

if __name__ == '__main__':
    # fixup_bad_gps()
    # fixup_bad_outputs()
    # fixup_bad_gps()
    fixup_bad_gps()
    fixup_bad_outputs()
