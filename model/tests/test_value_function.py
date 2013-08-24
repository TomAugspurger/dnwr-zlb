from __future__ import division

import cPickle
import os
import unittest

import nose
import numpy as np
from numpy.testing.decorators import slow
from scipy.optimize import fminbound
from scipy.stats import lognorm

from ..gen_interp import Interp
from ...model import gen_interp
from ..value_function import bellman, u_, get_rigid_output

from ..helpers import ss_output_flexible, ss_wage_flexible
from ..run_value_function import run_one

np.random.seed(42)


class testFunctions(unittest.TestCase):

    def test_utility(self):
        wage = 2
        # eta = 1.5
        # gamma = 0.5
        expected = 0.32409060804383427
        result = u_(wage, shock=.25, eta=2.5, gamma=0.5, aggL=4)
        self.assertAlmostEqual(expected, result)

    def test_ss_out(self):
        params = {'gamma': [0.5, '_'], 'eta': [1.5, '_'],
                  'sigma': [.02, '_']}
        expected = ((.5 / 1.5) ** (.5 / 1.5) *
                    (1 / (np.exp(-.5 * 1.5 * 1.5 / 2 * .02 ** 2))) ** (.5 / 1.5)
                    )
        result = ss_output_flexible(params)
        self.assertAlmostEqual(expected, result)

    def test_ss_wage(self):
        params = {'gamma': [0.5, '_'], 'eta': [2.5, '_'], 'sigma': [0.2, '_']}
        shock = 1
        agg_l = ss_output_flexible(params)  # probably bad
        expected = ((2.5 / 1.5) ** (.5 / 3) * shock ** ((0.5) / (3)) *
                    agg_l ** (1.5 / 3))
        self.assertAlmostEqual(expected, ss_wage_flexible(params, shock=shock))

        expected_shock = ((2.5 / 1.5) ** (.5 / 3) * 2 ** (0.5 / 3) *
                          agg_l ** (1.5 / 3))
        self.assertAlmostEqual(expected_shock,
                               ss_wage_flexible(params, shock=2))


class TestValueFunction(unittest.TestCase):

    def test_flexible(self):
        ss_w = 1.0041753592911187  # from ..vf_iteration.ss_wage_flexible
        h_ = lambda x: -1 * u_(x)
        xopt = fminbound(h_, .5, 3)
        self.assertAlmostEqual(xopt, ss_w, places=5)

    def test_bellman_smoke(self):

        params = {
            "lambda_": [0.8, "degree of rigidity"],
            "pi": [0.02, "target inflation"],
            "eta": [2.5, "elas. of subs among labor types"],
            "gamma": [0.5, "frisch elas. of labor supply"],
            "wl": [0.3, "wage lower bound"],
            "wu": [2.0, "wage upper bound"],
            "wn": [400, "wage grid point"],
            "beta": [0.97, "disount factor. check this"],
            "tol": [10e-6, "error tolerance for iteration"],
            "sigma": [0.2, "standard dev. of underlying normal dist"]}

        w_grid = np.linspace(0.1, 4, 5)

        sigma = params['sigma'][0]
        mu = -(sigma ** 2) / 2
        params['mu'] = mu, 'mean of underlying nomral distribution.'
        # ln_dist = lognorm(sigma, scale=np.exp(-(sigma) ** 2 / 2))
        # trunc = truncate_distribution(norm(loc=mu, scale=sigma), .05, .95)
        np.random.seed(42)
        sigma = params['sigma'][0]
        mu = -(sigma ** 2) / 2
        ln_dist = lognorm(sigma, scale=np.exp(-(sigma) ** 2 / 2))
        params['full_ln_dist'] = ln_dist, "Frozen lognormal distribution."
        ln_dist_lb = ln_dist.ppf(.05)
        ln_dist_ub = ln_dist.ppf(.95)
        zn = 5
        z_grid = np.linspace(ln_dist_lb, ln_dist_ub, zn)
        w0 = Interp(w_grid, -w_grid + 4)

        Tv, ws, vals = bellman(w0, params=params, w_grid=w_grid,
                               z_grid=z_grid)
        expected_y = np.array([3.76732473, 3.66307253, 2.56966702,
                               1.70810273, 0.91432274])
        np.testing.assert_almost_equal(expected_y, Tv.Y)

    @slow
    def test_ss_wage_flexible(self):
        X = np.array([ 0.70541378, 0.73668086, 0.76794794, 0.79921503, 0.83048211,
        0.86174919, 0.89301627, 0.92428335, 0.95555043, 0.98681752,
        1.0180846 , 1.04935168, 1.08061876, 1.11188584, 1.14315293,
        1.17442001, 1.20568709, 1.23695417, 1.26822125, 1.29948833,
        1.33075542, 1.3620225 ])

        Y = np.array([ 0.94743673, 0.95430996, 0.96094425, 0.96735712, 0.97356423,
        0.97957954, 0.98541565, 0.99108389, 0.99659452, 1.00195689,
        1.00717949, 1.01227009, 1.01723582, 1.02208323, 1.02681834,
        1.03144672, 1.03597354, 1.04040356, 1.04474122, 1.04899067,
        1.05315574, 1.05724005])

        expected = Interp(X, Y)

        z_grid = np.linspace(0.70541378068079674, 1.3620224972427708, 22)
        w_grid = np.linspace(0.40000000000000002, 3.5, 40)

        sigma = .2
        mu = -(sigma ** 2) / 2
        ln_dist = lognorm(sigma, scale=np.exp(-(sigma) ** 2 / 2))

        params = {
            "lambda_": [0.0, "degree of rigidity"],
            "pi": [0.02, "target inflation"],
            "eta": [2.5, "elas. of subs among labor types"],
            "gamma": [0.5, "frisch elas. of labor supply"],
            "wl": [0.3, "wage lower bound"],
            "wu": [2.0, "wage upper bound"],
            "wn": [400, "wage grid point"],
            "beta": [0.97, "disount factor. check this"],
            "tol": [10e-6, "error tolerance for iteration"],
            "sigma": [0.2, "standard dev. of underlying normal dist"],
            'z_grid': (z_grid, 'a'),
            'w_grid': (w_grid, 'a'),
            'full_ln_dist': (ln_dist, 'a'),
            'mu': (mu, 'mean of underlying nomral distribution.')}

        res_dict = run_one(params)
        actual = res_dict['ws']
        np.testing.assert_almost_equal(actual.Y, expected.Y, 5)
    # @slow
    # def test_ss_output_flex_close(self):
    #     expected = 0.85049063822172699
    #     with open('tests/files/flex_ws.pkl', 'r') as f:
    #         os.chdir('..')
    #         flex_ws = cPickle.load(f)
    #         os.chdir('tests')

    #     with open('tests/files/flex_g.pkl', 'r') as f:
    #         os.chdir('..')
    #         gf = cPickle.load(f)
    #         os.chdir('tests')

    #     sigma = 0.2
    #     z_grid = np.linspace(0.70541378068079674, 1.3620224972427708, 22)
    #     w_grid = np.linspace(0.40000000000000002, 3.5, 40)

    #     ln_dist = lognorm(sigma, scale=np.exp(-(sigma) ** 2 / 2))

    #     params = {'eta': (2.5, 'a'),
    #               'sigma': (sigma, 'a'),
    #               'gamma': (0.5, 'a'),
    #               'lambda_': (0, 'a'),
    #               'z_grid': (z_grid, 'a'),
    #               'w_grid': (w_grid, 'a'),
    #               'full_ln_dist': (ln_dist, 'a')
    #               }
    #     actual = get_rigid_output(flex_ws, params, flex_ws, gf)
    #     self.assertAlmostEquals(actual, expected)

if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
