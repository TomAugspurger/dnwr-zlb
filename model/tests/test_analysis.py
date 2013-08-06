from __future__ import division

import os
import unittest

import nose
import numpy as np

from ..analyze_run import read_output

np.random.seed(42)


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
            result = read_output(fnames)
        finally:
            [os.remove(fname) for fname in fnames]
        self.assertDictEqual(expected, result)


if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
