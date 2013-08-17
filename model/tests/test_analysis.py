from __future__ import division

import os
import pickle
import unittest

import nose
import numpy as np
from numpy.testing import assert_equal, assert_allclose

import pandas as pd
from pandas.util.testing import (assert_frame_equal,
                                 assert_panel_equal,
                                 )

from ..analyze_run import read_output, make_panel, get_utils
from ..gen_interp import Interp
from ..helpers import sample_path

np.random.seed(42)


def assert_interp_equal(left, right):
    assert left.X == right.X
    assert left.Y == right.Y
    assert left.kind == right.kind


class TestAnalysis(unittest.TestCase):

    def test_output_name_parse(self):
        fnames = ['rigid_output_0005_0497368421053_.txt',
                  'rigid_output_0005_09_.txt',
                  'thisshouldnotmatch.txt',
                  '.DS_Store']
        for fname in fnames:
            with open(fname, 'w') as f:
                f.write('1.0')

        expected = {(.005, .497368421053): 1.0,
                    (.005, .9): 1.0}
        try:
            result = read_output(fnames, kind='rigid_output')
        finally:
            [os.remove(fname) for fname in fnames]
        self.assertDictEqual(expected, result)

    def test_read_vf(self):
        fnames = ['vf_001_0139473684211.pkl',
                  'vf_005_09.pkl',
                  'thisshouldnotmatch.txt']
        vf = Interp([0, 1], [0, 1])
        for fname in fnames:
            with open(fname, 'w') as f:
                pickle.dump(vf, f)

        expected = {(.01, .139473684211): vf,
                    (.05, .9): vf}
        try:
            result = read_output(fnames, kind='vf')
        finally:
            [os.remove(fname) for fname in fnames]
        [assert_interp_equal(vf, result[x]) for x in result]
        self.assertEquals(sorted(expected.keys()), sorted(result.keys()))

    def test_sample_path(self):
        X = np.array([0.70541378, 0.73997213, 0.77453049, 0.80908884, 0.84364719,
                      0.87820555, 0.9127639, 0.94732226, 0.98188061, 1.01643896,
                      1.05099732, 1.08555567, 1.12011402, 1.15467238, 1.18923073,
                      1.22378908, 1.25834744, 1.29290579, 1.32746414, 1.3620225])

        Y = np.array([0.94741585, 0.95498567, 0.96226472, 0.96926453, 0.97600736,
                      0.98250305, 0.98877408, 0.99483511, 1.00069849, 1.0063769,
                      1.01188147, 1.01722231, 1.02240867, 1.02744901, 1.03235118,
                      1.03712205, 1.04177051, 1.046307, 1.05074639, 1.05508846])
        ws = Interp(X, Y, kind='cubic')
        key = (0.005, 0.05)
        seed = 42
        params = {'mu': (-0.020000000000000004, 'mean.'),
                  'sigma': [0.2, u'standard dev.'],
                  'lambda_': [key[1], 'a']}
        exp_pths = np.array([[0.99713702],
                             [1.00947703],
                             [1.00472714],
                             [1.00188703],
                             [0.99245093]])

        exp_shocks = np.array([[0.96075536],
                               [1.03577164],
                               [1.006286],
                               [0.98902403],
                               [0.93358932]])

        pths, shocks = sample_path(ws, params, lambda_=key[1], nperiods=5,
                                   seed=seed)
        assert_allclose(pths, exp_pths)
        assert_allclose(shocks, exp_shocks)

    def test_make_pan(self):
        X = np.array([0.70541378, 0.73997213, 0.77453049, 0.80908884, 0.84364719,
                      0.87820555, 0.9127639, 0.94732226, 0.98188061, 1.01643896,
                      1.05099732, 1.08555567, 1.12011402, 1.15467238, 1.18923073,
                      1.22378908, 1.25834744, 1.29290579, 1.32746414, 1.3620225])

        Y = np.array([0.94741585, 0.95498567, 0.96226472, 0.96926453, 0.97600736,
                      0.98250305, 0.98877408, 0.99483511, 1.00069849, 1.0063769,
                      1.01188147, 1.01722231, 1.02240867, 1.02744901, 1.03235118,
                      1.03712205, 1.04177051, 1.046307, 1.05074639, 1.05508846])
        ws = Interp(X, Y, kind='cubic')
        key = (0.005, 0.05)
        params = {'mu': (-0.020000000000000004, 'mean.'),
                  'sigma': [0.2, u'standard dev.'],
                  'lambda_': [key[1], 'a']}
        wpan, span = make_panel({key: ws}, params, pairs=[key], nseries=1,
                                nperiods=5)

        x = np.array([0.997137, 1.009477, 1.004727, 1.001887, 0.992451])
        s = np.array([0.96075536, 1.03577164, 1.006286, 0.98902403,
                      0.93358932])

        expceted_w = pd.Panel({key: pd.DataFrame({0: x})})
        expected_s = pd.Panel({key: pd.DataFrame({0: s})})
        assert_panel_equal(wpan, expceted_w)
        assert_panel_equal(span, expected_s)

    def test_utils(self):
        x = np.array([0.997137, 1.009477, 1.004727, 1.001887, 0.992451])
        s = np.array([0.96075536, 1.03577164, 1.006286, 0.98902403,
                      0.93358932])
        key = (0.005, 0.05)
        wpan = pd.Panel({key: pd.DataFrame({0: x})})
        span = pd.Panel({key: pd.DataFrame({0: s})})

        key = (0.005, 0.05)

        idx = pd.MultiIndex.from_tuples((key,))
        out_ser = pd.DataFrame({0: 2.02410628}, index=idx)

        actual = get_utils(wpan, span, out_ser=out_ser)
        expected = np.array([[0.37540230271201075]])
        assert_equal(expected, actual)

if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
