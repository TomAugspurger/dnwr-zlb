from __future__ import division

import json
import unittest

import numpy as np

from ..helpers import (load_params, truncated_draw, ss_output_flexible,
                       ss_wage_flexible)


class TestJson(unittest.TestCase):
    def setUp(self):
        # figure out relative filepaths
        with open('../parameters.json') as f:
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


class TestLoadParams(unittest.TestCase):

    def test_mean(self):
        params = load_params('../parameters.json')
        ln_dist = params['full_ln_dist'][0]
        np.random.seed(42)
        first = 0.99971395195248147
        second = ln_dist.rvs(10000).mean()
        self.assertEquals(first, second)

    def test_truncated_norm(self):
        params = load_params('../parameters.json')
        np.random.seed(42)
        expected = -0.02126563210945432
        actual = truncated_draw(params, .05, .95, kind='norm', size=10).mean()
        self.assertEquals(expected, actual)

    def test_ss_output_flexible(self):
        params = {'eta': (2.5, 'a'),
                  'gamma': (0.5, 'b'),
                  'sigma': (0.2, 'c')}

        expected = 0.85049063822172699
        actual = ss_output_flexible(params)
        self.assertEquals(expected, actual)

    def test_ss_wage_flexible(self):
        params = {'eta': (2.5, 'a'),
                  'gamma': (0.5, 'b'),
                  'sigma': (0.2, 'c')}

        actual = ss_wage_flexible(params, shock=1)
        expected = 1.0041753592911187
        self.assertEquals(expected, actual)
