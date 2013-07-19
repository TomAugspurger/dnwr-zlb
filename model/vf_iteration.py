from __future__ import division

import json

import numpy as np
from scipy.stats import lognorm
#-----------------------------------------------------------------------------
with open('parameters.json') as f:
    params = json.load(f)

wl = params['wl']     # wage lower bound
wu = params['wu']     # wage upper bound
wn = params['wn']     # wage grid point

w_grid = np.linspace(wl, wu, wn)
pi_grid = np.linspace(0, 2.5, .1)
pibar = 2.0  # ss inflation

#-----------------------------------------------------------------------------
# Setup the Distribution
# If log x is normally distributed with mean mu and variance sigma**,
# then x is log-normally distributed with shape paramter sigma and
# scale parameter exp(mu).
sigma = params['sigma']
mu = - (sigma ** 2) / 2

z_dist = norm(loc=mu, scale=sigma)  # z_dist ~ lognormally. E[exp(z)] = 1


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
    return stats.norm(loc=mu, scale=sigma, a=a, b=b)

if __name__ == '__main__':
    pass
