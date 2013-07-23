"""
A clean implementation of the value funciton.

v(w) = (1 - lambda_) * (u(today | w' >= 0) + beta * v(w')) +
            lambda_  * (u(today | w' >= w) + beta * v(w'))

"""
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.optimize import fminbound
from scipy import absolute as abs

from vf_iteration import truncate_normal
from lininterp import LinInterp
#-----------------------------------------------------------------------------
np.random.seed(42)
# Parameters

beta = .96
lambda_ = .8  # Their high value
eta = 2.5

sigma = 0.2
mu = -(sigma ** 2) / 2

trunc = truncate_normal(norm(loc=mu, scale=sigma), .05, .95)
shocks = np.sort(np.exp(trunc.rvs(1000)))

grid = np.linspace(0.1, 4, 100)


def maximizer(h_, a, b):
    return float(fminbound(lambda x: -h_(x), a, b))


def maximum(h_, a, b):
    return float(h_(fminbound(lambda x: -h_(x), a, b)))


def u_(wage, shock=1, eta=2.5, gamma=0.5, aggL=0.85049063822172699):
    utility = (wage ** (1 - eta) -
              ((gamma / (gamma + 1)) * shock *
              (wage ** (-eta) * aggL) ** ((gamma + 1) / gamma)))
    return utility

#-----------------------------------------------------------------------------


def bellman(w, u_fn, grid=grid, lambda_=0.8, shock=shocks, pi=2.0,
            argmax=False):
    """
    Operate on the bellman equation. Returns the policy rule or
    the value function (interpolated linearly) at each point in grid.

    Parameters
    ----------

    w : callable value function (probably instance of LinInterp from last iter)
    u_fn : The period utility function to be maximized. Omega in DH2013
    lambda : float. Degree of wage rigidity. 0 = flexible, 1 = fully rigid
    pi : steady-state (for now) inflation level.  Will be changed.

    Returns
    -------

    Tv : The next iteration of the value function v. Instance of LinInterp.
    """
    vals = []
    w_max = grid[-1]
    for y in grid:
        h_ = lambda x: np.mean(u_fn(x, shock)) + beta * w((x / (1 + pi)))
        if argmax:
            m1 = maximizer(h_, 0, w_max)
            m2 = maximizer(h_, y, w_max)
        else:
            m1 = maximum(h_, 0, w_max)
            m2 = maximum(h_, y, w_max)
        vals.append((1 - lambda_) * m1 + lambda_ * m2)
    return LinInterp(grid, np.array(vals))


def get_iterates(w0, maxiter=100, argmax=False, grid=grid, lambda_=0.8, pi=2.0,
                 shock=1):
    """
    Generator for bellman()

    Parameters
    ----------

    init_v : Initial guess for value funtion / policy rule (argmax=True)
    argmax : Bool
        True :  policy rule
        False : value function (default)


    see bellman doc for rest.

    Returns
    -------

    stream of Tv.
    """
    iters = [w0]
    for i in range(maxiter):
        Tv = bellman(iters[i], u_, grid=grid, lambda_=lambda_, shock=shock,
                     pi=pi, argmax=argmax)
        iters.append(Tv)
        yield Tv


def get_iters_takewhile(w0, tol, maxiter=1000, grid=grid, lambda_=0.8,
                        pi=2.0, shock=1, argmax=False):
    """
    Wrapper for get_iterators.

    Parameters
    ----------

    tol : Maximum distance between two succesive iterations.

    for the rest see bellman.

    Returns
    -------

    Tv: instance of LinInterp.
    """
    e = 1
    previous = w0
    gen = get_iterates(w0, grid=grid, lambda_=0.8, pi=2.0, shock=1,
                       n_iters=maxiter, argmax=False)

    for i in range(maxiter):
        print('iteration: {}, error: {}.'.format(i, e))
        Tv = next(gen)
        e = np.max(np.abs(Tv - previous))
        if e < tol:
            return Tv
        previous = Tv


#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# The follow is some cool stuff. No gauruntees on it working/being tested.

def get_cool_stuff():
    w0 = LinInterp(grid, 1 / grid)
    h_ = lambda x: -1 * u_(x)  # minimize this  call u_ for actual value.
    grid = np.linspace(0.1, 4, 100)

    iters = get_iterates(w0)


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(grid, maxers)
    ax2 = plt.twinx(ax)
    ax2.plot(grid[1:], u_(grid[1:]), color='r')

    #--------------------------------------------------------------------------
    xopt, neg_fval, _, _ = fminbound(h_, .5, 3, full_output=True)
    grid2 = np.linspace(0.6, 2, 1500)

    plt.plot(plgrid2, [u_(x) for x in grid2])

    # YAYAAYYAYAAYA
    xopt == ss_wage_flexible(params)  # almost!  w/in 6 sigfig.
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------


def get_diffs(w):
    diffs = []
    w_max = grid[-1]
    for y in grid:
        m1 = maximizer(h_, 0, w_max)
        m2 = maximizer(h_, y, w_max)
        diffs.append((m1, m2))

    plt.figure()
    ax = plt.plot(grid, map(lambda x: u_(x[0]) - u_(x[1]), diffs))
    return ax, diffs


def plot_hours_and_utility_over_shocks(shocks):
    h_ = lambda x, shock: -1 * u_(x, shock=shock)
    ws = np.array([fminbound(h_, 0, 3, args=(x,)) for x in shocks])
    us = [u_(x, shock=y) for x, y in zip(ws, shocks)]
    ax = plt.plot(shocks, ws, label='hours', )
    ax = plt.plot(shocks, us, label='utils')
    plt.legend()
    return ax
