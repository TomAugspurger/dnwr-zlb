"""
Several helper functions for use in value function
iteration.

API CHANGES:

    trunc, ln_dist, fine_grid, grid > w_grid
    shock > z_grid; fine_shock > z_grid_fine

"""

from __future__ import division

import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fminbound
from scipy import stats

np.random.seed(42)
#-----------------------------------------------------------------------------
# Optimizatoin
#----------------------------------------------------------------------------


def maximizer(h_, a, b, args=()):
    return float(fminbound(h_, a, b, args=args))


def maximum(h_, a, b, args=()):
    return -1 * float(h_(fminbound(h_, a, b, args=args), *args))

#----------------------------------------------------------------------------
# Parameters
#----------------------------------------------------------------------------


def load_params(pth='parameters.json'):
    with open(pth) as f:
        params = json.load(f)

    wl = params['wl'][0]
    wu = params['wu'][0]
    wn = params['wn'][0]
    w_grid = np.linspace(wl, wu, wn)
    w_grid_fine = np.linspace(wl, wu, 10000)
    params['w_grid'] = w_grid, 'Wage support.'
    params['w_grid_fine'] = w_grid_fine, 'Finer wage support'

    sigma = params['sigma'][0]
    mu = -(sigma ** 2) / 2
    params['mu'] = mu, 'mean of underlying nomral distribution.'

    ln_dist = stats.lognorm(sigma, scale=np.exp(-(sigma) ** 2 / 2))
    params['full_ln_dist'] = ln_dist, "Frozen lognormal distribution."
    ln_dist_lb = ln_dist.ppf(.05)
    ln_dist_ub = ln_dist.ppf(.95)
    params['ln_dist_lb'] = ln_dist_lb, "lower bound of lognorm dist."
    params['ln_dist_ub'] = ln_dist_ub, "upper bound of lognorm dist."

    zn = params['zn'][0]
    z_grid = np.linspace(ln_dist_lb, ln_dist_ub, zn)
    params['z_grid'] = z_grid, "Trucnated support of shocks."
    params['z_grid_fine'] = (np.linspace(ln_dist_lb, ln_dist_ub, zn),
                             "Finer shock support,")

    return params


def truncated_draw(params, lower, upper, kind='lognorm', size=1000):
    """
    Return a new normal distribution that is truncated given a
    lower upper tail in probabilities.

    Parameters
    ----------

    params : dict
    lower : probability chopped off lower end
    upper : probability chopper off the top
    kind : one of:
                -norm
                -lognorm
    size : How many draws to take. Default 1000.

    Returns
    -------

    array.

    Example
    -------
    This:
        >>>stats.lognorm(sigma, scale=np.exp(-(sigma) ** 2 / 2))
    is equivalent to
        >>>stats.norm(loc=mu, scale=sigma)

    where mu is -(sigma**2) / 2.
    """
    mu, sigma = params['mu'][0], params['sigma'][0]
    n_dist = stats.norm(mu, sigma)
    a, b = n_dist.ppf([.05, .95])
    truncated = stats.truncnorm(a, b, loc=mu, scale=sigma).rvs(size)
    if kind == 'lognorm':
        return np.exp(truncated)
    elif kind == 'norm':
        return truncated
    else:
        raise ValueError("Type must be one of 'norm' or 'lognorm'.")


def clean_shocks(new_shocks, calibrated_shocks):
    new_shocks[new_shocks < calibrated_shocks[0]] = calibrated_shocks[0]
    new_shocks[new_shocks > calibrated_shocks[-1]] = calibrated_shocks[-1]
    return new_shocks

#----------------------------------------------------------------------------
# Steady State Solutions
#----------------------------------------------------------------------------


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
    agg_l = ss_output_flexible(params)

    if shock is None:
        shock = params['z_grid'][0]

    wage = ((eta / (eta - 1)) ** (gamma / (gamma + eta)) *
            shock ** (gamma / (gamma + eta)) *
            agg_l ** ((1 + gamma) / (gamma + eta)))
    return wage

#----------------------------------------------------------------------------
#


def sample_path(ws, params, nseries=50):
    """
    Given a wage schedule, simulate the sample path of length nseries.
    """

    ln_dist = params['ln_dist'][0]
    shocks = clean_shocks(ln_dist.rvs(nseries), params['shock'][0])

    line = plt.plot(ws(shocks))
    return line
