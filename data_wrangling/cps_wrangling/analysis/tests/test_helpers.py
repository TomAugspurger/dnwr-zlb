import unittest

import numpy as np
import pandas as pd
import pandas.util.testing as tm

from data_wrangling.cps_wrangling.analysis.helpers import (bin_others,
                                                           date_parser,
                                                           filter_panel)


class TestHelpers(unittest.TestCase):

    def test_bin_others(self):
        s = pd.Series([1, 2, 3, 4, 1, 1, 1, 2, 2, 2, 3])
        others = [3, 4]
        result = bin_others(s, others)
        expected = pd.Series([1, 2, 'other', 'other', 1, 1, 1, 2, 2, 2, 'other'])
        tm.assert_series_equal(result, expected)

        with np.testing.assert_raises(ValueError):
            bin_others(s, [3, 4, 12])


class TestReadPanel(unittest.TestCase):

    # def setUp(self):
    #     self.store = pd.HDFStore('tst.h5')
    #     df = pd.DataFrame({'month': range(1, 13)})
    #     wp = pd.Panel({'i'})

    def test_date_parser(self):
        result = date_parser('2010_01').strftime('%Y_%m')
        expected = '2010_01'
        assert result == expected

        result = date_parser('2010-01').strftime('%Y_%m')
        assert result == expected

        result = date_parser('/m2010_01').strftime('%Y_%m')
        assert result == expected

        result = date_parser('m2010_01').strftime('%Y_%m')
        assert result == expected

    def test_filter_panel(self):
        dfa = pd.DataFrame({4: [1, 25, 30, 60, 75]})
        dfs = pd.DataFrame({4: [1, 1, 2, 1, 2]})
        wp = pd.Panel({'age': dfa, 'sex': dfs})

        result = filter_panel(wp, 'age')
        eage = pd.DataFrame({4: [25, 30, 60]}, index=[1, 2, 3])
        expected = pd.Panel({'age': eage, 'sex': dfs.loc[[1, 2, 3]]})

        tm.assert_panel_equal(result, expected)

        # age and sex
        result = filter_panel(wp, 'age', 'sex')
        expected = pd.Panel({'age': pd.DataFrame({4: [25, 60]}, index=[1, 3]),
                             'sex': pd.DataFrame({4: [1, 1]}, index=[1, 3])})

        tm.assert_panel_equal(result, expected)
