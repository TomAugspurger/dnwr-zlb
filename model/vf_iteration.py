from __future__ import division

from functools import partial
import json

import numpy as np
from scipy.stats import norm, lognorm
#-----------------------------------------------------------------------------


def load_params(pth='parameters.json'):
    with open(pth) as f:
        params = json.load(f)

    return params

params = load_params()
VALUE, DESCRIPTION = 0, 1

wl = params['wl'][VALUE]     # wage lower bound
wu = params['wu'][VALUE]     # wage upper bound
wn = params['wn'][VALUE]     # wage grid point

w_v = np.linspace(wl, wu, wn)
pi_grid = np.linspace(0, 2.5, 10)
pibar = 2.0  # ss inflation


#-----------------------------------------------------------------------------
# Setup the Distribution
# If log x is normally distributed with mean mu and variance sigma**,
# then x is log-normally distributed with shape paramter sigma and
# scale parameter exp(mu).


def setup_shocks():
    sigma = params['sigma'][VALUE]
    mu = - (sigma ** 2) / 2

    z_dist = norm(loc=mu, scale=sigma)  # z_dist ~ lognormally. E[exp(z)] = 1
    pass


def truncate_normal(original, lower, upper):
    """
    Return a new normal distribution that is truncated given a
    lower upper tail in probabilities.

    Parameters
    ----------

    original: frozen normal distribution with loc and scale specified.
    lower : probability chopped off lower end
    upper : probability chopper off the top

    Returns
    -------

    frozen_normal

    Example
    -------
    z_dist = norm(loc=mu, scale=sigma)
    trunc = truncate_normal(z_dist, .05, .95)
    """
    a, b = original.ppf(lower), original.ppf(upper)
    mu, sigma = original.mean(), original.std()
    return norm(loc=mu, scale=sigma, a=a, b=b)


def ut_c(cons):
    return np.log(cons)


def ut_l(wage, shock, agg_L, params):
    """
    Utillity from labor part of utility funciton.

    Parameters
    ----------

    wage: float. real wage for i at time t
    shock: float. idiosyncratic shock for i
    agg_L: float. aggregate labor
    params: dict. dict of params

    Returns
    -------

    float: utility
    """
    eta = params['eta'][0]
    gamma = params['gamma'][0]

    utility = (wage ** (1 - eta) -
              ((gamma / (gamma + 1)) * shock *
              (wage ** (-eta) * agg_L) ** ((gamma + 1) / gamma)))
    return utility


def ss_output_flexible(params):
    eta = params['eta'][0]
    gamma = params['gamma'][0]
    sigma = params['sigma'][0]

    zt = np.exp(-0.5 * (eta * (1 + gamma)) / (gamma + eta) * sigma ** 2)
    return (((eta - 1) / eta) ** (gamma / (1 + gamma)) *
            (1 / zt) ** (gamma / (1 + gamma)))


def ss_wage_flexible(params, shock=None):
    """
    Given paraemters, get the ss wage.

    Parameters
    ----------

    params: dict
    shock: float; Draw from the distribution.  defaults to the mean.

    Returns
    -------

    wage: float
    """
    eta = params['eta'][0]
    gamma = params['gamma'][0]
    sigma = params['sigma'][0]
    agg_l = ss_output_flexible(params)

    shock = shock or 1

    wage = ((eta / (eta - 1)) ** (gamma / (gamma + eta)) *
            shock ** (gamma / (gamma + eta)) *
            agg_l ** ((1 + gamma) / (gamma + eta)))
    return wage


def iter_vf(params):
    VALUE, DESCRIPTION = 0, 1

    wl = params['wl'][VALUE]     # wage lower bound
    wu = params['wu'][VALUE]     # wage upper bound
    wn = params['wn'][VALUE]     # wage grid point

    w_grid = np.linspace(wl, wu, wn)
    pi_grid = np.linspace(0, 2.5, 10)
    pibar = params['pibar'][0]  # ss inflation
    #-------------------------------------------------------------------------



if __name__ == '__main__':
    pass
