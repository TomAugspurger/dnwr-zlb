"""
Quality check on the two panels: `full` and `earn`.
"""
import json

import pandas as pd


class Checker(object):

    def __init__(self, settings, kind):

        if isinstance(str, settings):
            with open(settings, 'r') as f:
                settings = json.load(f)

        self.settings = settings
        self.canon = self.get_canon()
        self.list_to_check = self.get_list_to_check()
        self.kind = kind
        # self.cps_store = pd.HDFStore(settings["store_path"])
        self.get_cps_store = pd.get_store(settings["store_path"])
        self.get_full_store = pd.get_store(settings["panel_path"])
        self.get_earn_store = pd.get_store(settings["earn_store_path"])

    def get_canon(self):
        """
        Treat August 2013's columns as the One True Representation.
        """

    def get_list_to_check(self):
        pass

    def __call__(self):
        return {'full': self._call_full, 'earn': self._call_earn}[self.kind]

    def _call_full(self):
        pass

    def _call_earn(self):
        pass


def check_columns():
    pass


def main():
    with open('settings.txt', 'r') as f:
        settings = json.load(f)

    full_check = Checker(settings, kind='full')
    full_check()

    earn_check = Checker(settings, kind='earn')
    earn_check()

if __name__ == '__main__':
    main()
