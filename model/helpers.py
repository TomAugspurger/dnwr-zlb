from __future__ import division

import json

import numpy as np

from scipy.optimize import fminbound
from scipy.stats import norm, lognorm

np.random.seed(42)
#-----------------------------------------------------------------------------
#----------------------------------------------------------------------------


def maximizer(h_, a, b, args=()):
    return float(fminbound(h_, a, b, args=args))


def maximum(h_, a, b, args=()):
    return -1 * float(h_(fminbound(h_, a, b, args=args), *args))

#----------------------------------------------------------------------------


def load_params(pth='parameters.json'):
    with open(pth) as f:
        params = json.load(f)

    grid = np.linspace(0.1, 4, 100)
    fine_grid = np.linspace(.1, grid[-1], 10000)

    sigma = params['sigma'][0]
    mu = -(sigma ** 2) / 2
    params['mu'] = mu, 'mean of underlying nomral distribution.'
    trunc = truncate_distribution(norm(loc=mu, scale=sigma), .05, .95)

    shock = np.sort(np.exp(trunc.rvs(30)))
    fine_shock = np.sort(np.exp(trunc.rvs(1000)))  # need to deal with endoints
    fine_shock = clean_shocks(fine_shock, shock)

    params['shock'] = shock, 'Random sample from lognomral'
    params['grid'] = grid, 'Wage grid'
    params['fine_grid'] = fine_grid, 'fine_grid'
    params['fine_shock'] = fine_shock, 'fine_shock'

    ln_dist = truncate_distribution(
        lognorm(sigma, scale=np.exp(-(sigma)**2 / 2)), .05, .95)

    params['ln_dist'] = ln_dist, 'Frozen log-normal distribution'
    return params


#-----------------------------------------------------------------------------
# Setup the Distribution
# If log x is normally distributed with mean mu and variance sigma**,
# then x is log-normally distributed with shape paramter sigma and
# scale parameter exp(mu).

def truncate_distribution(original, lower, upper):
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
    trunc = truncate_distribution(z_dist, .05, .95)
    """
    a, b = original.ppf(lower), original.ppf(upper)
    mu, sigma = original.mean(), original.std()
    return norm(loc=mu, scale=sigma, a=a, b=b)


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
        shock = 1

    wage = ((eta / (eta - 1)) ** (gamma / (gamma + eta)) *
            shock ** (gamma / (gamma + eta)) *
            agg_l ** ((1 + gamma) / (gamma + eta)))
    return wage


def clean_shocks(new_shocks, calibrated_shocks):
    new_shocks[new_shocks < calibrated_shocks[0]] = calibrated_shocks[0]
    new_shocks[new_shocks > calibrated_shocks[-1]] = calibrated_shocks[-1]
    return new_shocks
