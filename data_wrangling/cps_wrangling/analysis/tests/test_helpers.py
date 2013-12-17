import unittest

import numpy as np
import pandas as pd
import pandas.util.testing as tm

from data_wrangling.cps_wrangling.analysis import helpers
from data_wrangling.cps_wrangling.analysis.helpers import (bin_others,
                                                           date_parser,
                                                           filter_panel,
                                                           make_df_ss)
from data_wrangling.cps_wrangling.analysis.make_to_long import quarterize


class TestHelpers(unittest.TestCase):

    def test_bin_others(self):
        s = pd.Series([1, 2, 3, 4, 1, 1, 1, 2, 2, 2, 3])
        others = [3, 4]
        result = bin_others(s, others)
        expected = pd.Series([1, 2, 'other', 'other', 1, 1, 1, 2, 2, 2, 'other'])
        tm.assert_series_equal(result, expected)

        with np.testing.assert_raises(ValueError):
            bin_others(s, [3, 4, 12])

    def test_make_df_ss(self):

        idx = pd.DatetimeIndex(start='1996-01-01', periods=4, freq='QS', name='qmonth')
        # idx = pd.MultiIndex.from_tuples(idx0, )
        df = pd.DataFrame({'age': [9, 9, 11, 11], 'sex_d': [0, 0, 1, 1],
                           'race_d': [0, 0, 1, 1], 'married_d': [0, 0, 1, 1],
                           'edu_bin': [0, 1, 3, 4], 'expr': [1, 2, 3, 4],
                           'expr_2': [1, 4, 9, 16], 'expr_3': [1, 8, 27, 64],
                           'expr_4': [1, 16, 81, 256]}, index=idx)
        params = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1], index=df.columns)
        df['either_history'] = [0, 0, 1, 1]
        lrprod = pd.DataFrame({'durable': [.1, .5, 1, 2],
                               'nondurable': [.1, .5, 1, 2],
                               'all_': [.1, .5, 1, 2]}, index=idx)
        result = make_df_ss(df, params, lrprod)
        levels = [[u'from_e', u'from_n', u'all_'],
                  [u'wage_index', u'all_', u'durable', u'nondurable', u'Q_1', u'Q_2', u'Q_3', u'Q_4']]
        labels = [[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
                  [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]]
        cols = pd.MultiIndex(levels=levels, labels=labels)
        expected = pd.DataFrame(np.array([[27,    .1,   .1,
                                           0.1,   1.0,   0,
                                           0,   0, np.nan,
                                           0.1,   .1,   .1,
                                           1.0,   0,   0,
                                           0,   27,   .1,
                                           0.1,   .1,   1.0,
                                           0,   0,   0],
                                          [np.nan,   .5,   .5,
                                           .5,   0,   1.0,
                                           0,   0, np.nan,
                                           .5,   .5,   .5,
                                           0,   1.0,   0,
                                           0,   95,   .5,
                                           .5,   .5,   0,
                                           1.0,   0,   0],
                                          [np.nan,   1.0,   1.0,
                                           1.0,   0,   0,
                                           1.0,   0,   221,
                                           1.0,   1.0,   1.0,
                                           0,   0,   1.0,
                                           0,   221,   1.0,
                                           1.0,   1.0,   0,
                                           0,   1.0,   0],
                                          [np.nan,   2,   2,
                                           2,   0,   0,
                                           0,   1.0, np.nan,
                                           2,   2,   2,
                                           0,   0,   0,
                                           1.0, np.nan,   2,
                                           2,   2,   0,
                                           0,   0,   1.0]]), index=idx, columns=cols)
        tm.assert_frame_equal(result, expected)


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

    def test_read_to_long_replace_variable_hours(self):
        df = pd.DataFrame({'hours': [-4, 1, -4, 1],
                           'actual_hours': [1, 1, 2, 1]},
                          index=['a', 'b', 'c', 'd'])

        expected = pd.DataFrame({'hours': [1, 1, 2, 1],
                                 'actual_hours': [1, 1, 2, 1]},
                                index=['a', 'b', 'c', 'd'])

        result = helpers.replace_variable_hours(df)
        tm.assert_frame_equal(result, expected)

    def test_fix_edu_bug(self):
        df = pd.DataFrame({'edu': [31, 32, 321, 411]})
        result = helpers.fix_edu_bug(df)
        expected = pd.DataFrame({'edu': [31, 32, 32, 41]})
        tm.assert_frame_equal(result, expected)

    def test_replace_catagorical(self):
        df = pd.DataFrame({'sex': [1, 2],
                           'race': [1, 2],
                           'married': [1, 4],
                           'labor_status': [1, 2],
                           'industry': [1, 3],
                           'occupation': [1, 7],
                           'edu': [31, 35],
                           'flow': [1, 3],
                           'history': [0, 1]})
        df_ = df.copy()
        expected = pd.DataFrame({'sex': ['male', 'female'],
                                 'race': ['White Only', 'Black Only'],
                                 'married': ["MARRIED, CIVILIAN SPOUSE PRESENT", "WIDOWED"],
                                 'labor_status': ['employed', 'absent'],
                                 'industry': ["Agriculture", "Mining"],
                                 'occupation': ["Management", "Legal"],
                                 'edu': ["LESS THAN 1ST GRADE", "9TH GRADE"],
                                 'flow': ['ee', 'en'],
                                 'history': ['employed', 'not_employed']})
        # full
        result = helpers.replace_categorical(df_)
        tm.assert_frame_equal(result, expected)

        for k in df.columns:
            df_ = df.copy()
            r1 = helpers.replace_categorical(df_, kind=k)
            ef = df.copy()
            ef[k] = expected[k]
            tm.assert_frame_equal(r1, ef)

        # inverse
        inv = helpers.replace_categorical(expected, inverse=True)
        tm.assert_frame_equal(inv, df)

        s = pd.DataFrame({"flow": ['ee', 'eu', 'en', 'ue', 'uu', 'un', 'ne',
                                   'nu', 'nn']})
        expected = pd.DataFrame({"flow": [1, 2, 3, 4, 5, 6, 7, 8, 9]})
        result = helpers.replace_categorical(s, kind='flow', inverse=True)
        tm.assert_frame_equal(result, expected)

    def test_quarterize(self):
        df = pd.DataFrame({'year':     [2012., 2012., 2012., 2012., 2012.],
                           'month':    [1, 2, 5, 8, 11],
                           'HRHHID':   [1, 1, 2, 2, 2],
                           'HRHHID2':  [1, 2, 1, 2, 2],
                           'PULINENO': [1, 2, 1, 2, 3]})
        df = df.set_index(['HRHHID', 'HRHHID2', 'PULINENO'])
        result = quarterize(df)
        expected = pd.DataFrame({'year': [2012, 2012, 2012, 2012, 2012],
                                 'month': [1, 2, 5, 8, 11],
                                 'quarter': [1, 1, 2, 3, 4],
                                 'HRHHID': [1, 1, 2, 2, 2],
                                 'HRHHID2': [1, 2, 1, 2, 2],
                                 'PULINENO': [1, 2, 1, 2, 3]})
        expected['qmonth'] = pd.to_datetime(["2012-01-01", "2012-01-01", "2012-04-01",
                                             "2012-07-01", "2012-10-01"])
        expected = expected.set_index(['qmonth', 'HRHHID', 'HRHHID2', 'PULINENO'])
        result = result[['month', 'year', 'quarter']]
        expected = expected[['month', 'year', 'quarter']]
        tm.assert_frame_equal(result, expected)

    def test_make_demo_dummies(self):
        df = pd.DataFrame({'race': [1, 2, 6, 10, 11, 12],
                           'sex': [1, 2, 2, 2, 2, 2],
                           'married': [0, 1, 2, 2, 2, 2]})
        expected = pd.DataFrame({'race_d': [0, 1, 1, 1, 1, 1],
                                 'sex_d': [0, 1, 1, 1, 1, 1],
                                 'married_d': [0, 1, 1, 1, 1, 1]})
        expected = expected[['race_d', 'sex_d', 'married_d']]
        result = helpers.make_demo_dummies(df)
        tm.assert_frame_equal(result, expected)

    def test_bin_education(self):
        df = pd.DataFrame({'edu': [31, 32, 33, 34, 35, 36, 37, 38,
                                   39, 40, 41, 42, 43, 44, 45, 46]})
        expected = pd.Series([0, 0, 0, 0, 0, 0, 0, 0,
                              1, 2, 3, 3, 4, 5, 5, 5], name='edu_d')
        result = helpers.bin_education(df)
        tm.assert_series_equal(result, expected)
