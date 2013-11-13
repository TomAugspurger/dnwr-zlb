import unittest

import numpy as np
import pandas as pd
import pandas.util.testing as tm

from data_wrangling.cps_wrangling.analysis.helpers import bin_others

class TestHelpers(unittest.TestCase):

    def test_bin_others(self):
        s = pd.Series([1, 2, 3, 4, 1, 1, 1, 2, 2, 2, 3])
        others = [3, 4]
        result = bin_others(s, others)
        expected = pd.Series([1, 2, 'other', 'other', 1, 1, 1, 2, 2, 2, 'other'])
        tm.assert_series_equal(result, expected)

        with np.testing.assert_raises(ValueError):
            bin_others(s, [3, 4, 12])
