"""
Cleaning up the inevible bugs without rerunning uncesscessary parts.

Try to keep a log of runs here.

Date       | Method
--------------------------
2013-08-11 | fixup_bad_gps
"""
import datetime
import shutil

import analyze_run as ar
import pickle
from helpers import load_params
from run_value_function import get_wage_distribution


def fixup_bad_gps():
    """
    Got flat wage distributions due to solo valid grid points.

    See commit `e5ace15` and tag `simulation3`.
    """
    params = load_params()
    params['results_path/'] = 'results/', 'a'
    all_files = ar.get_all_files(params)
    gps = ar.read_output(all_files, kind='gp')

    def _get_bad_gps(gp, cutoff=.5):
        """Ones whose max - min is less than, say, .5"""
        if gps[gp].Y[-1] - gps[gp].Y[0] < cutoff:
            return True
        else:
            return False

    bad_gps_keys = filter(_get_bad_gps, gps)
    bad_gps = {key: gps[key] for key in bad_gps_keys}

    for key in bad_gps:
        piname, lambda_ = [str(x).replace('.', '') for x in key]
        out_name = 'results/gp_' + piname + '_' + lambda_ + '.pkl'
        shutil.copy2(out_name, 'results/replaced_results/')
        _replace(key, out_name, all_files, params)
        print("Fixed {}".format(key))

    with open('results/fixup_notice.txt', 'w') as f:
        x = str(datetime.datetime.now())
        f.write("FIXED gps AT {}".format(x))


def _replace(key, out_name, all_files, params):
    wses = ar.read_output(all_files, kind='ws')
    ws = wses[key]
    gp = get_wage_distribution(ws, params)
    with open(out_name, 'w') as f:
        pickle.dump(gp, f)

if __name__ == '__main__':
    fixup_bad_gps()
