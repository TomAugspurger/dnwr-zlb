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
# Parameters

beta = .96
lambda_ = .8  # Their high value
eta = 2.5

sigma = 0.2
mu = -(sigma ** 2) / 2

trunc = truncate_normal(norm(loc=mu, scale=sigma), .05, .95)
shocks = trunc.rvs(1000)

grid = np.linspace(0.1, 4, 100)

# two choices.  Impose lower bound in utility function or with maximizer.
# I think w/ maximizer is better so use maximizer() and call bellman
# with w as the upper bound, b.

# actually minimizer
def maximizer_wlb(h_, a, b, wlb):
    return float(fminbound(h_, a, b, (wlb,)))


def maximizer(h_, a, b):
    return float(fminbound(lambda x: -h_(x), a, b))


def u_(wage, shock=1, eta=2.5, gamma=0.5, aggL=0.85049063822172699):
    utility = (wage ** (1 - eta) -
              ((gamma / (gamma + 1)) * shock *
              (wage ** (-eta) * aggL) ** ((gamma + 1) / gamma)))
    return utility


def u_min(wage, wlb=0, shock=1, eta=2.5, gamma=0.5, aggL=0.85049063822172699):
    """ Should be same as u_ but flipped for max/min"""
    if wage < wlb:
        return 10000
    else:
        utility = (wage ** (1 - eta) -
                  ((gamma / (gamma + 1)) * shock *
                  (wage ** (-eta) * aggL) ** ((gamma + 1) / gamma)))
        return -1 * utility

# def u_vec(wage, wlb, grid, shock=1, eta=2.5, gamma=0.5, aggL=2):
#     utility = (wage ** (1 - eta) -
#               ((gamma / (gamma + 1)) * shock *
#               (wage ** (-eta) * aggL) ** ((gamma + 1) / gamma)))
#     utility[grid <= wlb] = -10000
#     return utility

# def v_(vals, grid, w_l):
#     """To be iterated upon by T.  Must operate on vectors"""
#     utility = np.sqrt(vals)
#     utility[grid <= w_l] = -100000
#     return utility

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------


def get_greedy(w, wlb=0):
    vals = []
    for y in grid:
        vals.append(maximizer(u_min, 0, y, wlb) + w(y))
    return LinInterp(grid, vals)

# sigma = get_greedy()


def T_(sigma, v_):
    """Operator"""
    vals = []
    for y in grid:
        Tw_y = np.mean(u_(sigma(y), shock=shocks)) + beta * np.mean(v_(sigma(y / 2)))
        vals.append(Tw_y)
    return LinInterp(grid, vals)


def get_value(sigma, v_):
    """Policy rule and value funciton"""
    tol = 1e-2
    i = 0
    maxiter = 10000
    while 1 and i <= maxiter:
        new_v = T_(sigma, v_)
        err = np.max((new_v(grid) - v_(grid)))
        if err < tol:
            return new_v
        v_ = new_v
        i += 1
        print(i)

#-----------------------------------------------------------------------------


def bellman(w, u_fn, lambda_=0.8):
    # This one is better; restricts the space.
    vals = []
    w_max = grid[-1]
    for y in grid:
        w_prev = w.Y[y]
        maxer = (1 - lambda_) * maximizer(u_fn, 0, w_max) + lambda_ * maximizer(u_fn, w_prev, w_max)
        # val = -1 * u_fn(maxer)
        vals.append(maxer)
    return LinInterp(grid, np.array(vals))


def bellman_fixup(w, u_fn, lambda_=0.8, pi=2):
    # Building off bellman.  Want to be passing in w as well (for the future value).
    # mean isn't really doing anything right now, but will be in future.
    # passing x to w vs. u_ may also be wrong.  is it the input? choice wagE?
    # it's returing the same for every iteration.  Not sure what's going on.
    vals = []
    w_max = grid[-1]
    for y in grid:
        w_prev = w.Y[y]
        h_ = lambda x: u_(x) + beta * np.mean(w(u_(x)))
        # maxer = (1 - lambda_) * maximizer(u_fn, 0, w_max) + lambda_ * maximizer(u_fn, w_prev, w_max)
        # val = -1 * u_fn(maxer)
        m1 = maximizer(h_, 0, w_max)
        m2 = maximizer(h_, y, w_max)
        vals.append((1 - lambda_) * m1 + lambda_ * m2)
    return LinInterp(grid, np.array(vals))


# def bellman_wlb(w, u_min, wlb=0):
#     vals = []
#     w_max = grid[-1]
#     lambda_ = 0.8
#     for y in grid:
#         maxer = (1 - lambda_) * maximizer(u_min, 0, w_max, wlb) + lambda_ * maximizer(u_min, 0, w_max, y)
#         val = -1 * u_min(maxer, wlb)
#         vals.append(val)
#     return LinInterp(grid, np.array(vals))


def get_iterates(init_v, lambda_=0.8, n_iters=30):
    iters = [init_v]
    for i in range(n_iters):
        Tv = bellman(iters[i], h_, lambda_)
        iters.append(Tv)
    iters.pop(0)
    return iters

w0 = LinInterp(grid, np.sqrt(grid))
h_ = lambda x: -1 * u_(x)  # minimize this  call u_ for actual value.
grid = np.linspace(0.1, 4, 100)

iters = get_iterates(w0)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(grid, maxers)
ax2 = plt.twinx(ax)
ax2.plot(grid[1:], u_(grid[1:]), color='r')

#-----------------------------------------------------------------------------
xopt, neg_fval, _, _ = fminbound(h_, .5, 3, full_output=True)
grid2 = np.linspace(0.6, 2, 1500)

plt.plot(plgrid2, [u_(x) for x in grid2])

# YAYAAYYAYAAYA
xopt == ss_wage_flexible(params)  # almost!  w/in 6 sigfig.
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# What is the difference?
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
