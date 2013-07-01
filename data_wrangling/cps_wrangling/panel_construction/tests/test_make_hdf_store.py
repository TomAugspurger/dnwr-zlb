import os
import unittest
import json

import pandas as pd

from ..make_hdf_store import runner, dedup_cols, pre_process


class TestLoadSettings(unittest.TestCase):

    def setUp(self):
        settings_path = 'info.txt'
        f = open(settings_path)
        self.settings = json.load(f)

    def test_load_succed(self):
        self.assertIsInstance(self.settings, dict)

    def test_get_dd(self):
        dd = self.settings["month_to_dd"]["198901"]
        expected = "jan1989"
        self.assertEqual(expected, dd)
 
