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


def runner(params):

    def vf_iter(w_v, vf, tv, maxit, beta, return_fctn):
        n = len(vf)
        # pr = np.zeros(n)
        for i in range(maxit):
            for w_ind in range(n):
                wlb = w_v[w_ind]
                ar = return_fctn(w_v, wlb)
                tv[w_ind] = np.max(ar[w_ind, :] + beta * vf)

            diff = np.max(np.abs(tv - vf))
            print(i, diff)
            if diff < tol:
                break

            for t in range(n):
                vf[t] = tv[t]

        return vf

    VALUE, DESCRIPTION = 0, 1

    wl = params['wl'][VALUE]     # wage lower bound
    wu = params['wu'][VALUE]     # wage upper bound
    wn = params['wn'][VALUE]     # wage grid point
    tol = params['tol'][VALUE]
    beta = params['beta'][VALUE]

    lambda_ = params['lambda_'][VALUE]  # probability of DNWR

    w_array = np.linspace(wl, wu, wn)
    w_grid = np.tile(w_array, (wn, 1)).T
    # pi_grid = np.linspace(0, 2.5, 10)
    # pibar = params['pibar'][0]  # ss inflation

    vf = np.ones(wn) * .05
    vf_p = np.zeros(wn)  # new value function
    pr = np.zeros(wn)    # policy rule

    utl_part = partial(ut_l, shock=1, agg_L=ss_output_flexible(params),
                       params=params)

    utility = map(utl_part, w_array)
    e = 1
    max_iter = 1000
    iteration = 0
    while e > tol and iteration < max_iter:
        for i, wage in enumerate(w_array):
            temp = utility[i] + beta * vf
            ind = np.argmax(temp)
            pr[i] = w_array[ind]
            if ind in [0, w_array - 1] and iteration > 0:
                print("Boundry Warning")

            temp = utility[i] + beta * vf
            vf_p[i] = temp[ind]

        e = np.max(np.abs(vf - vf_p))
        iteration += 1
        vf = np.copy(vf)
        print("iteration: {}, error: {}".format(iteration, e))
    return vf, pr
    #-------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------


# def clean bellman(v):
#     Tv = []
#     for x


if __name__ == '__main__':
    pass
