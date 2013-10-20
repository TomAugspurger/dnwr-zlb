"""
Quality check on the two panels: `full` and `earn`.
"""
import json

import pandas as pd


class Checker(object):

    def __init__(self, settings, kind):

        if isinstance(settings, str):
            with open(settings, 'r') as f:
                settings = json.load(f)

        self.settings = settings
        self.kind = kind
        #----------------------------------------------------------------------
        # Load stores
        self.cps_store = pd.HDFStore(settings["store_path"])
        self.panel = pd.HDFStore(settings["panel_path"])
        #-----------------------------------------------------------------------
        # Todo List with reference
        self.canon = self.get_canon()
        self.all_keys = filter(lambda x: x.startswith('m'),
                               dir(self.cps_store.root.monthly.data))
        self.panel_keys = self.panel.keys()

    def get_canon(self):
        """
        Treat August 2013's columns as the One True Representation.
        """
        return self.cps_store.select('/monthly/data/m2013_08').columns

    def __call__(self):
        return {'panel': self._call_panel, 'earn': self._call_earn}[self.kind]()

    def _call_panel(self):
        """
        Setter for self.outlist, a list of dicts of lists
        """
        gen = self.get_panel_overlap()
        self.out_list = list(gen)

    def write_panel(self):
        """ IO """
        with open("panel_check.json", 'w') as f:
            json.dump(self.out_list, f)

    def get_panel_overlap(self):
        for k in self.panel_keys:
            cols = self.panel.select(k).minor_axis
            missing = (self.canon - cols).tolist()
            extra = (cols - self.canon).tolist()
            common = self.canon.intersection(cols).tolist()
            yield {"missing": missing, "extra": extra, "common": common}
            print("Yiedled {}".format(k))

    def _call_earn(self):
        raise NotImplementedError


def main():
    with open('settings.txt', 'r') as f:
        settings = json.load(f)

    full_check = Checker(settings, kind='panel')
    full_check()
    full_check.write()
    # earn_check = Checker(settings, kind='earn')
    # earn_check()

if __name__ == '__main__':
    main()
