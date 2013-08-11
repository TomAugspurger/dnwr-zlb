from __future__ import division

import os
import pickle
import unittest

import nose
import numpy as np

from ..analyze_run import read_output
from ..gen_interp import Interp

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


if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
