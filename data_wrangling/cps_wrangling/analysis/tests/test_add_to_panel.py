import unittest

import pandas as pd
import numpy as np
import pandas.util.testing as tm

from data_wrangling.cps_wrangling.analysis import add_to_panel

class TestAddHistory(unittest.TestCase):

    def setUp(self):

        self.test = pd.HDFStore('test_store.h5')
        # will be transposed
        status_frame = pd.DataFrame({'a': [1, 1, 1, 1, 1, 1, 1, 1],
                                     'b': [1, 2, 1, 2, 1, 1, 1, 3],
                                     'c': [3, 1, 1, 1, 3, 3, 3, 1],
                                     'd': [3, 1, 1, 3, 1, 1, 1, 5],
                                     'e': [5, 1, 1, 1, 3, 3, 3, 3],
                                     'f': [5, 1, 1, 5, 5, 5, 1, 1],
                                     'g': [5, 1, 1, 3, 1, 1, 6, 1],
                                     'h': [1, 1, 1, 3, 1, 1, 3, 1],
                                     'i': [1, 1, 1, 5, 1, 1, 1, 2]
                                     }, index=range(1, 9)).T
        self.wp = pd.Panel({'labor_status': status_frame})

    def test_history(self):
        wp = self.wp.copy()
        result = add_to_panel._add_employment_status_last_period(wp, 'unemployed',
                                                                 inplace=False)
        expected = pd.DataFrame([np.nan]).reindex_like(wp['labor_status'])
        expected.loc['a', [4, 8]] = False
        expected.loc['b', 4] = False
        expected.loc['c', [4, 8]] = True
        # expected.loc['e', 4] = np.NaN  #  donesn't match kind
        # expected.loc['f', 8] = np.Nan
        # expected.loc['g', 8] = np.Nan
        expected.loc['h', 8] = True
        expected.loc['i', 8] = False

        tm.assert_frame_equal(result, expected)

    def tearDown(self):

        import os
        self.test.close()
        os.remove('test_store.h5')
