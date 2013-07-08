import unittest

import pandas as pd

from ..march_supp_dd_parser import (item_identifier, item_parser,
                                              dict_constructor, checker,
                                              writer, month_item)

from ..generic_data_dictionary_parser import Parser

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

    def test_regex_no_trailing(self):
        s = 'H-HHTYPE    CHARACTER*001 .     (0069:0069)'


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


class TestParserClass(unittest.TestCase):

    def setUp(self):
        self.parser = Parser('/cpsbjan03.ddf', 'fakefile2')

    def test_formatter(self):
        s = 'H-MONTH     CHARACTER*002 .     (0038:0039)'.rstrip()
        m = self.parser.regex.match(s)
        expected = ('H-MONTH', 2, 38, 39)
        self.assertEqual(expected, self.parser.formatter(m))

    def test_regex_paddding_trailing_space(self):
        s = 'PADDING  CHARACTER*039          (0472:0600) '.rstrip()
        expected = ('PADDING', '039', '0472', '0600')
        self.assertEqual(expected, self.parser.regex.match(s).groups())

    def test_regex_paddding(self):
        s = 'PADDING  CHARACTER*039          (0472:0600)'.rstrip()
        expected = ('PADDING', '039', '0472', '0600')
        self.assertEqual(expected, self.parser.regex.match(s).groups())

    def test_store_name_basic(self):
        expected = 'jan2003'
        self.assertEqual(expected, self.parser.get_store_name())

    def test_aug05_regex_basic(self):
        ring = 'HRHHID          15     HOUSEHOLD IDENTIFIER   (Part 1)             (1 - 15)'.rstrip()
        self.parser.regex = self.parser.make_regex(style='aug2005')
        expected = ('HRHHID', '15', 'HOUSEHOLD IDENTIFIER   (Part 1)', '1', '15')
        self.assertEqual(expected, self.parser.regex.match(ring).groups())

    def test_aug05_style_with_parens_id(self):
        ring = 'HRHHID       15    HOUSEHOLD IDENTIFIER   (Part 1)            1 - 15'.rstrip()
        self.parser.regex = self.parser.make_regex(style='aug2005')
        expected = ('HRHHID', '15', 'HOUSEHOLD IDENTIFIER   (Part 1)', '1', '15')
        self.assertEqual(expected, self.parser.regex.match(ring).groups())

    def test_aug05_style_dont_pickup_first_number(self):
        ring = 'HRMONTH      2     MONTH OF INTERVIEW                      16-17'.rstrip()
        self.parser.regex = self.parser.make_regex(style='aug2005')
        expected = ('HRMONTH', '2', 'MONTH OF INTERVIEW', '16', '17')
        self.assertEqual(expected, self.parser.regex.match(ring).groups())

    def test_aug05_style_with_nums_in_description(self):
        ring = 'PRPTHRS      2     AT WORK 1-34 BY HOURS AT WORK           403 - 404'.rstrip()
        self.parser.regex = self.parser.make_regex(style='aug2005')
        expected = ('PRPTHRS', '2', 'AT WORK 1-34 BY HOURS AT WORK', '403', '404')
        self.assertEqual(expected, self.parser.regex.match(ring).groups())

    def tets_jan98_style_miss_first(self):
        ring = 'DATA        SIZE  BEGIN'.rstrip()
        self.parser.regex = self.parser.make_regex(style='jan1998')
        expected = None
        self.assertEqual(expected, self.parser.regex.match(ring))

    def test_jan98_basic(self):
        ring = 'D HRHHID     15      1'.rstrip()
        self.parser.regex = self.parser.make_regex(style='jan1998')
        expected = ('HRHHID', '15', '1')
        self.assertEqual(expected, self.parser.regex.match(ring).groups())

    def test_aug05_style_tabs(self):
        ring = 'HRHHID\t\t15\t\tHOUSEHOLD IDENTIFIER (Part 1)\t\t\t\t\t 1- 15'.rstrip()
        self.parser.regex = self.parser.make_regex(style='aug2005')
        expected = ('HRHHID', '15', 'HOUSEHOLD IDENTIFIER (Part 1)', '1', '15')
        self.assertEqual(expected, self.parser.regex.match(ring).groups())

    def test_may12_style_tabs(self):
        ring = 'HRHHID\t\t15\t\tHOUSEHOLD IDENTIFIER (Part 1)\t\t\t\t\t 1- 15\r\n'.rstrip()
        self.parser.regex = self.parser.make_regex(style='aug2005')
        expected = ('HRHHID', '15', 'HOUSEHOLD IDENTIFIER (Part 1)', '1', '15')
        self.assertEqual(expected, self.parser.regex.match(ring).groups())

    def test_formfeed(self):
        ring = 'PRDTIND2     2     DETAILED INDUSTRY RECODE - JOB 2        (474 - 475)'
        self.parser.regex = self.parser.make_regex(style='aug2005')
        expected = ('PRDTIND2', '2', 'DETAILED INDUSTRY RECODE - JOB 2', '474', '475')
        self.assertEqual(expected, self.parser.regex.match(ring).groups())
