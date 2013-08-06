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
        fnames = ['tst_rigid_output_000973684210526_.txt',
                  'tst_rigid_output_000736842105263_.txt',
                  'thisshouldnotmatch.txt',
                  '.DS_Store']
        for fname in fnames:
            with open(fname, 'w') as f:
                f.write('1.0')

        expected = {.00973684210526: 1.0,
                    .00736842105263: 1.0}
        try:
            result = read_output(fnames, kind='rigid_output')
        finally:
            [os.remove(fname) for fname in fnames]
        self.assertDictEqual(expected, result)

    def test_read_vf(self):
        fnames = ['vf_00476315789474.pkl',
                  'vf_005.pkl',
                  'thisshouldnotmatch.txt']
        vf = Interp([0, 1], [0, 1])
        for fname in fnames:
            with open(fname, 'w') as f:
                pickle.dump(vf, f)

        expected = {.0476315789474: vf,
                    .05: vf}
        try:
            result = read_output(fnames, kind='vf')
        finally:
            [os.remove(fname) for fname in fnames]
        [assert_interp_equal(vf, result[x]) for x in result]
        self.assertEquals(sorted(expected.keys()), sorted(result.keys()))


if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
