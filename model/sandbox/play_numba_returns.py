from __future__ import division

from functools import partial
import json

import numpy as np
from scipy.stats import norm, lognorm

try:
    from numba import jit, autojit
except ImportError:
    pass

#-----------------------------------------------------------------------------


def returns_numba(w_v, wlb=0, shock=1.0, eta=2.5, gamma=0.5, agg_L=.9):
    """
    currently works only for row 0.  all others untouched.
    Calculate returns for a value function.

    Parameters
    ----------

    w_v: vector of wages
    wlb: wage lower bound.  Choosing wage below this sucks.
    shock: float
    eta: float. paramter
    gamma: float. parameter
    agg_L: float aggregate labor. parameter.

    Returns
    -------

    len(w_v) x len(w_v) array of returns for (wage_today, wage_tommorow) returns
    """
    @autojit
    def gen_returns(ar, w_v, wlb, shock, eta, gamma, agg_L):
        for i in range(n):
            for j in range(n):
                    if w_v[j] > wlb:  # feasible.  Door for lower bound here.
                        ar[i, j] = (w_v[j] ** (1 - eta) -
                                    (gamma / (gamma + 1)) *
                                    shock *
                                    ((w_v[j]) ** (-eta) *
                                    agg_L) ** ((gamma + 1) / gamma))
        return ar

    n = len(w_v)
    ar = np.ones([n, n]) * -100000
    ar = gen_returns(ar, w_v, wlb, shock, eta, gamma, agg_L)
    return ar


def returns(w_v, wlb=0, shock=1.0, eta=2.5, gamma=0.5, agg_L=.9):
    """
    Calculate returns for a value function.

    Parameters
    ----------

    w_v: vector of wages
    wlb: wage lower bound.  Choosing wage below this sucks.
    shock: float
    eta: float. paramter
    gamma: float. parameter
    agg_L: float aggregate labor. parameter.

    Returns
    -------

    len(w_v) x len(w_v) array of returns for (wage_today, wage_tommorow) returns
    """
    n = len(w_v)
    ar = np.ones([n, n]) * -100000

    for i in range(n):

        for j in range(n):
                if w_v[j] > wlb:  # feasible.  Door for lower bound here.
                    ar[i, j] = (w_v[j] ** (1 - eta) -
                                (gamma / (gamma + 1)) *
                                shock *
                                ((w_v[j]) ** (-eta) *
                                agg_L) ** ((gamma + 1) / gamma))
    return ar
