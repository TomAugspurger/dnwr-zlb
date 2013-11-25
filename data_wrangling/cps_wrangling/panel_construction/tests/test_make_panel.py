import unittest

import pandas as pd
import pandas.util.testing as tm

from data_wrangling.cps_wrangling.panel_construction.make_panel import match_panel


class TestMakePanel(unittest.TestCase):

    def test_match_panel(self):
        idx1 = pd.MultiIndex.from_tuples([(1, 'one'), (1, 'two'),
                                          (2, 'one'), (2, 'two'),
                                          (3, 'one'), (3, 'two'),
                                          (4, 'one'), (4, 'two'),
                                          ('six', 'A')])
        idx2 = pd.MultiIndex.from_tuples([(1, 'one'), (1, 'two'),
                                          (2, 'one'), (2, 'two'),
                                          (3, 'one'), (3, 'two'),
                                          (4, 'one'), (4, 'two'),
                                          ('ten', 'A')])
        df1 = pd.DataFrame({'PRTAGE': [4, 4, 4, 4, 4, 4, 4, 4, 10],
                            'PESEX':  [1, 1, 1, 1, 1, 1, 0, 0, 10]}, index=idx1)
        df2 = pd.DataFrame({'PRTAGE': [5, 4, 9, 1, 4, 4, 4, 4, 10],
                            'PESEX':  [1, 1, 1, 1, 1, 0, 1, 0, 10]}, index=idx2)
        result = match_panel(df1, df2).sort_index()
        eidx = pd.MultiIndex.from_tuples([(1, 'one'), (1, 'two'),
                                          (3, 'one'), (4, 'two')])
        expected = pd.DataFrame({'PRTAGE': [5, 4, 4, 4],
                                 'PESEX': [1, 1, 1, 0]}, index=eidx)
        tm.assert_frame_equal(result, expected, check_dtype=False)
