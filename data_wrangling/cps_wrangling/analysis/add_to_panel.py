"""
Run during or after panel_constructoin.

Every toplevel function should have the api `(month, panel_store)`

where month is a 'YYYY_MM' str.
Modifies the panels on disk... so be careful.
"""
import pandas as pd
import numpy as np

from helpers import get_useful

def add_flows(month, panel_store):
    """
    Add the *montly* flows for each worker, for each month (2 :: 8).

    The flows are: ee, eu, en, ue, uu, un, ne, nu, nn/
    """

    def add_flow_series(s1, s2, categorical=False):
        if categorical:
            c1 = ['employed', 'absent']
            c2 = ['layoff', 'looking']
            c3 = ['retired', 'disabled', 'other']
        else:
            c1 = [1, 2]
            c2 = [3, 4]
            c3 = [4, 5, 6]

        ee = s1.isin(c1) & s2.isin(c1)
        eu = s1.isin(c1) & s2.isin(c2)
        en = s1.isin(c1) & s2.isin(c3)
        ue = s1.isin(c2) & s2.isin(c1)
        uu = s1.isin(c2) & s2.isin(c2)
        un = s1.isin(c2) & s2.isin(c3)
        ne = s1.isin(c3) & s2.isin(c1)
        nu = s1.isin(c3) & s2.isin(c2)
        nn = s1.isin(c3) & s2.isin(c3)

        flows = {"ee": ee, "eu": eu, "en": en, "ue": ue, "uu": uu, "un": un,
                 "ne": ne, "nu": nu, "nn": nn}

        s = pd.Series(np.nan, index=s1.index)
        for k, v in flows.iteritems():
            s.loc[v] = k
        return s

    def add_flows_panel(wp, inplace=True):
        """wrapper for add_flows.
        The flow in 2 is the flow *from* 1 *to* 2. 1 will contain NaNs.

        If not doing it inplace, assings with wp.loc[:, :, 'flow'] = returned
        """
        labor = wp['labor_status']
        d = {1: pd.Series(np.nan, labor[1].index)}
        for i in range(1, 8):
            s1 = labor[i]
            s2 = labor[i+1]
            d[i+1] = add_flow_series(s1, s2, categorical=False)
        d = pd.DataFrame(d)
        if inplace:
            wp['flow'] = d
        else:
            return d

    _wp = panel_store.select('m' + month)
    wp = get_useful(_wp.copy())
    try:
        add_flows_panel(wp, inplace=True)
        _wp['flow'] = wp['flow']
        _wp.to_hdf(panel_store, 'm' + month, append=False)
    except Exception as e:
        print("Skipping {}, because of {}".format(month, e))


def add_history(month, panel_store):
    """
    Add the 3 month history for every employee working.

    Return frame of bools:

    True if new hire (according to kind)
    False if existing "hire"
    NaN if not currently working or NaN

    """

    def add_employment_status_last_period(wp, kind, inplace=True):
        """
        Add the employment status for each worker to a panel.

        Employment status is over the last 4 months (MIS 1-4, MIS 5-8)

        `kind` is one of {'either', 'unemployed', 'nonemployed'}

        Assumes useful names

        1    EMPLOYED-AT WORK
        2    EMPLOYED-ABSENT
        3    UNEMPLOYED-ON LAYOFF
        4    UNEMPLOYED-LOOKING
        5    NOT IN LABOR FORCE-RETIRED
        6    NOT IN LABOR FORCE-DISABLED
        7    NOT IN LABOR FORCE-OTHER
        """
        ls = wp['labor_status']
        d = {'either': [3, 4, 5, 6, 7], 'unemployed': [3, 4],
             'nonemployed': [5, 6, 7]}

        # get employed and get those in that kind. That's youre subset.
        # select status for that subset, check isin again. Trues are True.
        # Falses are employed. Reindex. nans are other.

        # all since changes caught below
        employed_at_4 = ls.loc[:, 1:4].isin([1, 2]).all(1)
        employed_at_8 = ls.loc[:, 5:8].isin([1, 2]).all(1)

        kind_at_4 = ls.loc[:, 1:4].isin(d[kind]).any(1)
        kind_at_8 = ls.loc[:, 5:8].isin(d[kind]).any(1)

        sub4 = ls.loc[:, 1:4][kind_at_4 | employed_at_4]
        sub4 = sub4.isin(d[kind]).any(1)

        sub8 = ls.loc[:, 5:8][kind_at_8 | employed_at_8]
        sub8 = sub8.isin(d[kind]).any(1)

        emp = pd.concat([kind_at_4, kind_at_8], axis=1, keys=[4, 8]).reindex_like(ls)

        if inplace:
            wp[kind] = emp
        else:
            return emp

    _wp = panel_store.select('m' + month)
    wp = get_useful(_wp.copy())
    e_types = ['either', 'unemployed', 'nonemployed']

    try:
        # inplace
        [add_employment_status_last_period(wp, kind=x) for x in e_types]
        _wp['unemployed_history'] = wp['unemployed']
        _wp['nonemployed_history'] = wp['nonemployed']
        _wp['either_history'] = wp['either']
        _wp.to_hdf(panel_store, 'm' + month, append=False)

    except KeyError:
        return None
