from __future__ import division

import json
import unittest

import numpy as np

from ..helpers import (load_params, truncated_draw, ss_output_flexible,
                       ss_wage_flexible)


class TestLoadParams(unittest.TestCase):

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
