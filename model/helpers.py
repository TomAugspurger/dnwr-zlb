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


def truncated_draw(params, lower=.05, upper=.95, kind='lognorm', size=1000,
                   samples=1):
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
    samples: Number of independent draws of size size to take.
        i.e. samples is the number of individuals while
        size is the number of periods.

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
    truncated = stats.truncnorm(a, b, loc=mu, scale=sigma).rvs([size, samples])
    if kind == 'lognorm':
        return np.exp(truncated)
    elif kind == 'norm':
        return truncated
    else:
        raise ValueError("kind must be one of 'norm' or 'lognorm'.")


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


def ss_wage_flexible(params, shock=None, nperiods=30, nseries=1):
    """
    Given paraemters, get the ss wage.

    Parameters
    ----------

    params: dict
    shock: float;
    nperiods: int :: Lenght of shock series. Only used if shock is None
    nseries: int :: Number of independent series to sample.

    Returns
    -------

    wage: float or array of floats.
    """
    eta = params['eta'][0]
    gamma = params['gamma'][0]
    agg_l = ss_output_flexible(params)

    if shock is None:
        shock = truncated_draw(params, size=nperiods, samples=nseries)

    wage = ((eta / (eta - 1)) ** (gamma / (gamma + eta)) *
            shock ** (gamma / (gamma + eta)) *
            agg_l ** ((1 + gamma) / (gamma + eta)))
    return wage

#----------------------------------------------------------------------------
#


def sample_path(ws, params, lambda_=None, w0=.9, nseries=1, nperiods=1000,
                seed=None):
    """
    Given a wage schedule, simulate the sample path of length nseries.

    w0 is the initial wage now (a float) not the initial wage schedule.

    Added lambda_ as an arg for easy override of the dict lambda.

    seed allows for setting a seed by passing an int.

    Returns
    -------

    vals : array of wages chosen with shape nperiods x nseries
    shocks: shocks that generated those choices.  Same shape.
    """
    # TODO: Change to idio shock rather than same for everyone
    if seed:
        if isinstance(seed, int):
            np.random.seed(seed)
        else:  # Say a bool, defaults to 42.
            np.random.seed(42)

    shocks = truncated_draw(params, size=nperiods, samples=nseries)
    lambda_ = lambda_ or params['lambda_'][0]
    cannot_change_arr = np.random.uniform(0, 1, [nperiods, nseries]) < lambda_

    # initialize as empty.  Fill first row with values from w, the initial wage
    vals = np.empty([nperiods, nseries])
    w = np.ones_like(vals[0, :]) * w0
    vals[0] = w

    # To vectorize, we take nseries draws from unif and check if less than lamb
    # If greater, cannot_change is 0, so 0 * p2 is 0, so they always choose
    # ws(shock), i.e. they are free to choose whatever.

    for i, shock in enumerate(shocks):
        cannot_change = cannot_change_arr[i]
        p1 = ws(shock)
        p2 = cannot_change * np.amax([w, ws(shock)], axis=0)
        vals[i] = np.amax([p1, p2], axis=0)
        w = vals[i]

    return np.array(vals), shocks
