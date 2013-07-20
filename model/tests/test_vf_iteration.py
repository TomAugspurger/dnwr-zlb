import unittest
import json

import nose

from ..vf_iteration import (truncate_normal, ut_c, ut_l, ss_output_flexible,
                            ss_wage_flexible)

import numpy as np
from scipy.stats import norm


class TestJson(unittest.TestCase):
    def setUp(self):
        # figure out relative filepaths
        with open('./parameters.json') as f:
            params = json.load(f)
        self.params = params

    def test_dtypes(self):
        wl = self.params['wl'][0]
        self.assertTrue(isinstance(wl, int))

        wl_desc = self.params['wl'][1]
        self.assertTrue(isinstance(wl_desc, (str, unicode)))


class testFunctions(unittest.TestCase):

    def test_cons(self):
        consumption = 12
        expected = np.log(12)
        self.assertEquals(expected, ut_c(consumption))

    def test_labor(self):
        wage = 2
        shock = .25
        agg_L = 4
        params = {'gamma': [0.5, '_'], 'eta': [1.5, '_']}
        expected = (2**(-.5) -
                   ((.5 / 1.5) * (.25) *
                   (2 ** (-1.5) * 4) ** (1.5 / 0.5)))
        result = ut_l(wage, shock, agg_L, params)
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
        self.assertAlmostEqual(expected, ss_wage_flexible(params))

        expected_shock = ((2.5 / 1.5) ** (.5 / 3) * 2 ** (0.5 / 3) *
                          agg_l ** (1.5 / 3))
        self.assertAlmostEqual(expected_shock,
                               ss_wage_flexible(params, shock=2))


class TestDistribution(unittest.TestCase):

    def test_truncate(self):
        dist = norm()
        res = truncate_normal(dist, .05, .95)
        expected = -np.inf, np.inf
        result = res.ppf(0), res.ppf(1)
        self.assertEquals(expected, result)



if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
