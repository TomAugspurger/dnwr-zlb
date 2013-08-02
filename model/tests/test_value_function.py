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
            4.33389223,  4.33389223,  4.33389223,  4.33389223,  4.33389223,
            4.33389223,  4.33389223,  4.33389223,  4.33389223,  4.33389223,
            4.33389223,  4.33389223,  4.33389223,  4.33389223,  4.33389223,
            4.33389223,  4.33389223,  4.33389223,  4.33389223,  4.33389223,
            4.33389223,  4.3338514,   4.33212299,  4.32372061,  4.30681411,
            4.28364278,  4.2567212,   4.2277259,   4.19774997,  4.16750283,
            4.13744028,  4.10784947,  4.0789049,   4.05070584,  4.02330131,
            3.99670709,  3.97091714,  3.94591146,  3.92166144,  3.89813348,
            3.87529143,  3.85309828,  3.83151723,  3.81051246,  3.79004951,
            3.77009557,  3.75061961,  3.73159245,  3.71298671,  3.69477677,
            3.67693873,  3.65945031,  3.64229072,  3.62543919,  3.60888059,
            3.5925967,   3.57657187,  3.56079157,  3.5452422,   3.52991114,
            3.51478655,  3.49985741,  3.48511345,  3.47054502,  3.4561431,
            3.44189924,  3.42780554,  3.41385458,  3.4000394,   3.38635347,
            3.37279065,  3.35934401,  3.34601042,  3.33278361,  3.31965874,
            3.30663128,  3.29369692,  3.28085165,  3.26809159,  3.25541312,
            3.2428128,   3.23028738,  3.21783273,  3.20544806,  3.19312954,
            3.18087456,  3.16868065,  3.15654538,  3.14446548,  3.13244087,
            3.12046846,  3.10854643,  3.09667282,  3.08484485,  3.07306295,
            3.0613235,   3.049627,    3.03797008,  3.02635279,  3.01477365])

        np.testing.assert_almost_equal(expected_y, Tv.Y)

if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
