import unittest
import json

import pandas as pd
import pandas.util.testing as tm

from ..make_hdf_store import main, dedup_cols, pre_process, standardize_ids


class TestLoadSettings(unittest.TestCase):

    def test_standardize_ids(self):
        df = pd.DataFrame({'HRSAMPLE': ['A77', 'A75', 'Z76'],
                           'HRSERSUF': [-1, 'A', 'Z'],
                           'HUHHNUM': ['01', '02', '01']})
        expected = pd.Series([770001, 750102, 762601])
        result = standardize_ids(df)
        tm.assert_series_equal(result, expected)
