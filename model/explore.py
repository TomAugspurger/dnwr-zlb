import datetime
import itertools as it

import numpy as np
import matplotlib.pyplot as plt

from value_function import iter_bellman, g_p, get_rigid_output
from helpers import load_params
from gen_interp import Interp


def setup(wns=None, zns=None):
    if wns is None:
        wns = [100]
    if zns is None:
        zns = [100]

    dic = {}
    for wn, zn in it.product(wns, zns):
        params = load_params()
        wl = params['wl'][0]
        wu = params['wu'][0]
        w_grid = np.linspace(wl, wu, wn)
        params['w_grid'] = w_grid, 'Wage support.'

        ln_dist_lb, _ = params['ln_dist_lb']
        ln_dist_ub, _ = params['ln_dist_ub']
        ln_dist, _ = params['full_ln_dist']

        z_grid = np.linspace(ln_dist_lb, ln_dist_ub, zn)
        params['z_grid'] = z_grid, "Trucnated support of shocks."
        params['wn'] = wn, ' '
        params['zn'] = zn, ' '
        dic[(wn, zn)] = params

    return dic


class Results(object):
    def __init__(self, res_dict, paramses):
        self.res_dict = res_dict
        self.paramses = paramses
        self.VF = 0
        self.WS = 1
        self.REST = 2
        self.DELTA = 3
        self.GP = 4
        self.slices = {'vf': 0, 'ws': 1, 'rest': 2, 'delta': 3, 'gp': 4}

    def diff_vf(self, l_key, r_key, kind='vf'):
        left = self.res_dict[l_key][self.slices[kind]]
        right = self.res_dict[r_key][self.slices[kind]]
        w_grid_fine = self.paramses[(100, 100)]['w_grid_fine'][0]
        return plt.plot(w_grid_fine, left(w_grid_fine) - right(w_grid_fine),
                        label=(l_key))

    def get_gp(self):
        self.gps = {}
        for run in self.res_dict.keys():
            ws = self.res_dict[run][self.WS]
            w_grid = self.paramses[run]['w_grid'][0]
            g0 = Interp(w_grid, w_grid/4, kind='pchip')
            params = self.paramses[run]
            gp = g_p(g0, ws, params)
            self.gps[run] = gp

    # def diff_ws(self, l_key, r_key):
    #     left = self.res_dict[l_key][self.VF]
    #     right = self.res_dict[r_key][self.VF]

if __name__ == '__main__':
    wns = [100, 400]
    zns = [100, 400]

    paramses = setup(wns, zns)

    res_dict = {}
    for wn_zn, params in paramses.iteritems():
        w_grid = params['w_grid'][0]
        w0 = Interp(w_grid, -w_grid+28)
        strt = datetime.datetime.now()
        Tv, ws, rest = iter_bellman(w0, tol=0.1, strict=False, log=False,
                                    params=params)
        time_delta = datetime.datetime.now() - strt
        print(wn_zn, str(time_delta))

        w_max = w_grid[-1]
        g0 = Interp(w_grid, w_grid/w_max, kind='pchip')
        gp = g_p(g0, ws, params)
        res_dict[wn_zn] = (Tv, ws, rest, time_delta, gp)

    res = Results(res_dict, paramses)
