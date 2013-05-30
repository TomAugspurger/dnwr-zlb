import unittest

import pandas as pd

from ..generic_data_dictionary_parser import (item_identifier, item_parser,
                                              dict_constructor, checker,
                                              writer, month_item)

from pandas.util.testing import (assert_panel_equal, assert_frame_equal,
                                 assert_series_equal, assert_almost_equal)


class TestDDParser(unittest.TestCase):

    def test_march_identifier_basic(self, march=True):
        line = 'D '
        self.assertEqual(True, item_identifier(line, march))

    def test_identifier_none(self, march=True):
        line = '\t'
        self.assertEqual(None, item_identifier(line, march))

    def test_item_parser_length(self, march=True):
        s = 'D HRECORD     1      1  (1:1)'
        length = item_parser(s)
        cond = length >= 4
        self.assertEqual(True, cond)

    def test_regex_basic(self):
        s = 'H$CPSCHK    CHARACTER*001 .     (0001:0001)           ALL'
        expected = ('H$CPSCHK', '001', '0001', '0001')
        returned = month_item(s).groups()
        self.assertEqual(expected, returned)

    def test_regex_dash(self):
        s = 'H-MONTH     CHARACTER*002 .     (0038:0039)'
        expected = ('H-MONTH', '002', '0038', '0039')
        self.assertEqual(expected, month_item(s).groups())

    @unittest.skip('Skipping')
    def test_dict_constructor(self):
        pass

    def test_id_padding(self):
        s = 'PADDING       CHARACTER*001 .     (0467:0467)'
        expected = ('PADDING', '001', '0467', '0467')
        self.assertEqual(expected, month_item(s).groups())

    def test_item_parser_length_padding(self):
        s = 'PADDING       CHARACTER*001 .     (0467:0467)'
        length = item_parser(s)
        cond = length >= 4
        self.assertEqual(True, cond)

    def test_checker_one(self):
        df = pd.DataFrame({'start': [1, 2, 10], 'length': [1, 8, 2]})
        assert_frame_equal(df, checker(df)[0])

    def test_checker_multi(self):
        df = pd.DataFrame({'start': [1, 2, 10, 1, 2, 10], 'length': [1, 8, 2,
                                                                     1, 8, 2]})
        expected = [pd.DataFrame({'start': [1, 2, 10], 'length': [1, 8, 2]}),
                    pd.DataFrame({'start': [1, 2, 10], 'length': [1, 8, 2]},
                                 index=[3, 4, 5])]
        result = checker(df)
        assert_frame_equal(expected[0], result[0])
        assert_frame_equal(expected[1], result[1], check_names=False)
