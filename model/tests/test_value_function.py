from __future__ import division

import json
import unittest

import nose
import numpy as np
from numpy.testing.decorators import slow
from scipy.optimize import fminbound
from scipy.stats import norm

from ..cfminbound import ch_
from ..gen_interp import Interp
from ..value_function import bellman, u_

from ..helpers import (truncate_distribution, ss_output_flexible,
                       ss_wage_flexible)

np.random.seed(42)


class TestJson(unittest.TestCase):
    def setUp(self):
        # figure out relative filepaths
        with open('./parameters.json') as f:
            params = json.load(f)
        self.params = params

    def test_dtypes(self):
        wl = self.params['wl'][0]
        self.assertTrue(isinstance(wl, (float, int)))

        wl_desc = self.params['wl'][1]
        try:  # Python 2.7
            self.assertTrue(isinstance(wl_desc, basestring))
        except NameError:  # py3
            self.assertTrue(isinstance(wl_desc, str))


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


class TestDistribution(unittest.TestCase):

    def test_truncate(self):
        dist = norm()
        res = truncate_distribution(dist, .05, .95)
        expected = -np.inf, np.inf
        result = res.ppf(0), res.ppf(1)
        self.assertEquals(expected, result)


class TestValueFunction(unittest.TestCase):

    def test_flexible(self):
        ss_w = 1.0041753592911187  # from ..vf_iteration.ss_wage_flexible
        h_ = lambda x: -1 * u_(x)
        xopt = fminbound(h_, .5, 3)
        self.assertAlmostEqual(xopt, ss_w, places=5)

    @slow
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

        grid = np.linspace(0.1, 4, 100)

        sigma = params['sigma'][0]
        mu = -(sigma ** 2) / 2
        params['mu'] = mu, 'mean of underlying nomral distribution.'
        trunc = truncate_distribution(norm(loc=mu, scale=sigma), .05, .95)
        shock = np.sort(np.exp(trunc.rvs(30)))
        w0 = Interp(grid, -grid + 4)

        Tv, ws, vals = bellman(w0, params=params, u_fn=u_, grid=grid,
                               shock=shock)
        expected_y = np.array([
            3.77756583,  3.77756583,  3.77756583,  3.77756583,  3.77756583,
            3.77756583,  3.77756583,  3.77756583,  3.77756583,  3.77756583,
            3.77756583,  3.77756583,  3.77756583,  3.77756583,  3.77756583,
            3.77756583,  3.77756583,  3.77756583,  3.77756583,  3.77756583,
            3.77755226,  3.77546776,  3.76298984,  3.73654061,  3.69981807,
            3.65675965,  3.60995268,  3.56107198,  3.51121065,  3.46107811,
            3.41113017,  3.36165395,  3.31282399,  3.26473953,  3.21744961,
            3.17096999,  3.12529464,  3.08040356,  3.03626814,  2.9928548,
            2.95012738,  2.90804886,  2.86658245,  2.82569231,  2.78534398,
            2.74550467,  2.70614335,  2.66723082,  2.62873971,  2.5906444,
            2.552921,    2.5155472,   2.47850224,  2.44176365,  2.40531968,
            2.36915041,  2.33324022,  2.29757454,  2.26213981,  2.22692338,
            2.19191342,  2.15709892,  2.12246961,  2.08801586,  2.05372861,
            2.01959943,  1.9856204,   1.95178412,  1.91808362,  1.88451237,
            1.85106422,  1.81773055,  1.78451164,  1.7513995,   1.71838931,
            1.68547652,  1.65265686,  1.61992634,  1.58728103,  1.5547173,
            1.52223173,  1.48982107,  1.45747946,  1.42520954,  1.39300577,
            1.36086559,  1.32878654,  1.29676615,  1.2647994,   1.23288965,
            1.20103212,  1.16922513,  1.13746658,  1.10575196,  1.07408528,
            1.04245948,  1.01087856,  0.97933589,  0.94783396,  0.91637111])

        np.testing.assert_almost_equal(expected_y, Tv.Y)

if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
