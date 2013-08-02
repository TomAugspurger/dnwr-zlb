from __future__ import division

import unittest

import nose
import numpy as np
from numpy.testing.decorators import slow
from scipy.optimize import fminbound
from scipy.stats import norm

from ..lininterp import LinInterp, Interp
from ..value_function import bellman, u_
from test_vf_iteration import truncate_distribution

np.random.seed(42)


class TestValueFunction(unittest.TestCase):

    def test_flexible(self):
        ss_w = 1.0041753592911187  # from ..vf_iteration.ss_wage_flexible
        h_ = lambda x: -1 * u_(x)
        xopt = fminbound(h_, .5, 3)
        self.assertAlmostEqual(xopt, ss_w, places=5)

    @slow
    def test_bellman_smoke(self):
        grid = np.linspace(0.1, 4, 100)
        sigma = 0.2
        mu = -(sigma ** 2) / 2
        trunc = truncate_distribution(norm(loc=mu, scale=sigma), .05, .95)
        shock = np.sort(np.exp(trunc.rvs(30)))
        w0 = Interp(grid, -grid + 4)

        Tv, ws, vals = bellman(w0, u_, grid=grid, lambda_=.8, shock=shock)
        expected_y = np.array([
        3.74665947,  3.74665947,  3.74665947,  3.74665947,  3.74665947,
        3.74665947,  3.74665947,  3.74665947,  3.74665947,  3.74665947,
        3.74665947,  3.74665947,  3.74665947,  3.74665947,  3.74665947,
        3.74665947,  3.74665947,  3.74665947,  3.74665947,  3.74665947,
        3.74664792,  3.74463417,  3.73239643,  3.70625003,  3.66983646,
        3.62708705,  3.58058904,  3.53201732,  3.48246496,  3.4326414,
        3.38300242,  3.33383518,  3.28531418,  3.23753869,  3.19055775,
        3.1443871,   3.09902072,  3.05443862,  3.01061217,  2.96750779,
        2.92508935,  2.8833198,   2.84216236,  2.80158119,  2.76154183,
        2.7220115,   2.68295914,  2.64435559,  2.60617344,  2.56838711,
        2.53097267,  2.49390785,  2.45717186,  2.42074226,  2.38460727,
        2.34874697,  2.31314575,  2.27778904,  2.24266328,  2.20775582,
        2.17305483,  2.1385493,   2.10422896,  2.07008418,  2.0361059,
        2.00228569,  1.96861564,  1.93508832,  1.90169679,  1.86843451,
        1.83529533,  1.80227066,  1.76936072,  1.73655755,  1.70385633,
        1.67125251,  1.63874183,  1.60632027,  1.57398393,  1.54172917,
        1.50955257,  1.47745088,  1.44541826,  1.41345732,  1.38156251,
        1.3497313,   1.31796122,  1.28624979,  1.25459204,  1.22299126,
        1.19144269,  1.15994467,  1.12849508,  1.09708945,  1.06573173,
        1.03441492,  1.00314295,  0.97190926,  0.9407163,   0.90956239])

        np.testing.assert_almost_equal(expected_y, Tv.Y)

if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
