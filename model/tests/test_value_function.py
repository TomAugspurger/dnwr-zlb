from __future__ import division

import unittest

import nose
import numpy as np
from numpy.testing.decorators import slow
from scipy.optimize import fminbound
from scipy.stats import lognorm

from ..gen_interp import Interp
from ..value_function import bellman, u_, g_p

from ..helpers import ss_output_flexible, ss_wage_flexible

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

    def test_handle_solo(self):
        ws = Interp(np.array([0.70541378, 0.73997213, 0.77453049, 0.80908884,
                              0.84364719, 0.87820555, 0.9127639,  0.94732226,
                              0.98188061, 1.01643896, 1.05099732, 1.08555567,
                              1.12011402, 1.15467238, 1.18923073, 1.22378908,
                              1.25834744, 1.29290579, 1.32746414, 1.3620225]),
                    np.array([0.94864294, 0.95732392, 0.96597872, 0.9736925,
                              0.98001698, 0.98540488, 0.99011949, 0.99432335,
                              0.9981246,  1.00160187, 1.00480861, 1.00778999,
                              1.01057645, 1.01319409, 1.01566555, 1.01800755,
                              1.020233,  1.0223574,  1.02438508, 1.02632872]))

        w_grid = np.linspace(0.4, 3.5, 40)
        w_max = w_grid[-1]
        g = Interp(w_grid, w_grid/w_max, kind='pchip')
        params = {'lambda_': (0.855263157895, 'a'),
                  'pi': (.01, 'pi'),
                  'gamma': (0.5, "frisch elas. of labor supply"),
                  'sigma': (0.2, "standard dev. of underlying normal dist"),
                  'wn': (40, u'wage grid point'),
                  "wl": [0.4, "wage lower bound"],
                  "wu": [3.5, "wage upper bound"],
                  "w_grid": (w_grid, 'a')}
        np.random.seed(42)
        sigma = params['sigma'][0]
        ln_dist = lognorm(sigma, scale=np.exp(-(sigma) ** 2 / 2))
        params['full_ln_dist'] = ln_dist, "Frozen lognormal distribution."
        actual = g_p(g, ws, params)

        expected_X_mean = 1.9674999999999998
        expected_Y_mean = 0.1534990161757282

        self.assertEquals(expected_X_mean, actual.X.mean())
        self.assertEquals(expected_Y_mean, actual.Y.mean())

if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
