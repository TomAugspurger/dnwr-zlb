import unittest
import json

import nose

from ..vf_iteration import truncate_normal, ut_c, ut_l

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
        self.assertTrue(isinstance(wl_desc, unicode))


class testFunctions(unittest.TestCase):

    def test_cons(self):
        consumption = 12
        expected = np.log(12)
        self.assertEquals(expected, ut_c(consumption))

    def test_labor(self):
        hours = 2
        wage = 2
        shock = .25
        agg_L = 4
        params = {'gamma': [0.5, '_'], 'eta': [1.5, '_']}
        expected = 2**(-.5) - (.5 / 1.5) * (.25) * ((2 ** (-1.5) * 4) ** (.5 / 1.5))
        result = ut_l(wage, shock, agg_L, params)

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
