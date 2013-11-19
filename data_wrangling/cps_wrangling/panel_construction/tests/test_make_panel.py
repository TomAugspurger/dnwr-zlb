import unittest

import pandas as pd
import pandas.util.testing as tm

from data_wrangling.cps_wrangling.panel_construction.make_panel import huhhnum_to_hrhhid_index


class TestMakePanel(unittest.TestCase):

    def test_huhhnum_to_hrhhid(self):
        # yes, no, no
        idx1 = pd.MultiIndex.from_tuples([(1, 1, 1), (1, 1, 1), (2, 3, 1)])
        idx2 = pd.MultiIndex.from_tuples([(1, 11, 1), (1, 12, 1), (1, 13, 1)])

        df1 = pd.DataFrame({'timestamp': pd.to_datetime(3 * ['2003-02-01'])}, index=idx1)
        df2 = pd.DataFrame({'timestamp': pd.to_datetime(3 * ['2003-02-01'])}, index=idx2)

        df1.index.names = ['HRHHID', 'HRHHID2', 'PULINENO']
        df2.index.names = ['HRHHID', 'HRHHID2', 'PULINENO']
        result = huhhnum_to_hrhhid_index(df1, df2)
        expected = pd.DataFrame({'timestamp': pd.to_datetime(3 * ['2003-02-01'])},
                                index=pd.MultiIndex.from_tuples([(1, 1, 1),
                                                                 (1, 2, 1),
                                                                 (1, 3, 1)]))
        expected.index.names = ['HRHHID', 'HRHHID2', 'PULINENO']
        tm.assert_frame_equal(result, expected)
