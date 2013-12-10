import json
import unittest

import pandas as pd
import pandas.util.testing as tm
from pandas.util.testing import network


from bls import fetch_data, parse_data, make_data


class TestBLS(unittest.TestCase):

    @network
    def test_bls_quarter(self):
        data1 = json.dumps({"seriesid": ["PRS85006093", "PRS85006153"],
                            "startyear": "1994", "endyear": "2004"})
        result = parse_data(fetch_data(data1)).head(1)
        expected = pd.DataFrame({'PRS85006093': [69.303],
                                 'PRS85006153': [81.559]})
        expected.index = pd.to_datetime(['1994-01-01'])
        expected.index.name = 'stamp'
        expected.columns.name = 'series_id'
        tm.assert_frame_equal(result, expected)

    @network
    def test_bls_industry(self):
        data = make_data(["IPUUN8111__L000"])
        result = parse_data(fetch_data(data)).head(1)
        expected = pd.DataFrame([99.718], columns=["IPUUN8111__L000"])
        expected.index = pd.to_datetime(['2003-01-01'])
        expected.index.name = 'stamp'
        expected.columns.name = 'series_id'
        tm.assert_frame_equal(result, expected)

    def test_bls_split_years(self):
        result = list(make_data('PRS85006093', start="1994", end="2004"))
        expected = ['{"seriesid": "PRS85006093", "endyear": "2004", '
                    '"startyear": "1994"}']
        self.assertEqual(result, expected)

        result = list(make_data('PRS85006093', start="1994", end="2010"))
        expected = ['{"seriesid": "PRS85006093", "endyear": "2004", '
                    '"startyear": "1994"}',
                    '{"seriesid": "PRS85006093", "endyear": "2010", '
                    '"startyear": "2005"}']
        self.assertEqual(result, expected)
