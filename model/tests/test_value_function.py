from __future__ import division

import unittest

import nose
import numpy as np
from numpy.testing.decorators import slow
from scipy.optimize import fminbound
from scipy.stats import lognorm

from ..cfminbound import ch_
from ..gen_interp import Interp
from ..value_function import bellman, u_

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

    def test_ch_(self):
        expected = -2.3293269585464986
        grid = np.linspace(0.1, 4, 100)
        w0 = Interp(grid, -grid + 4)
        result = ch_(2.0, 2.0, w0, .02)
        self.assertEquals(expected, result)


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
            "sigma": [0.2, "std. dev. of log(Z) (shocks)"],
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

if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
