"""
A clean implementation of the value funciton.

v(w) = (1 - lambda_) * (u(today | w' >= 0) + beta * v(w')) +
            lambda_  * (u(today | w' >= w) + beta * v(w'))

"""
from __future__ import division

import itertools as it

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
pi = 2

grid = np.linspace(0.1, 4, 100)
sigma = 0.2
mu = -(sigma ** 2) / 2
trunc = truncate_normal(norm(loc=mu, scale=sigma), .05, .95)
shock = np.sort(np.exp(trunc.rvs(30)))


def maximizer(h_, a, b, args=()):
    return float(fminbound(h_, a, b, args=args))


def maximum(h_, a, b, args=()):
    return -1 * float(h_(fminbound(h_, a, b, args=args), *args))


def u_(wage, shock=1, eta=2.5, gamma=0.5, aggL=0.85049063822172699):
    utility = (wage ** (1 - eta) -
              ((gamma / (gamma + 1)) * shock *
              (wage ** (-eta) * aggL) ** ((gamma + 1) / gamma)))
    return utility

#-----------------------------------------------------------------------------


def bellman(w, u_fn=u_, grid=None, lambda_=0.8, shock=None, pi=2.0):
    """
    Differs from bellman by optimizing for *each* shock, rather than
    for the mean.  I think this is right since the agent observes Z_{it}
    before choosing w_{it}.

    Returns [(wage, shock, w*)]

    where w* is the convex combo of optimal for constrained vs. unconstrained
    case.  Currently that is 100 x 1000 x ?

    Operate on the bellman equation. Returns the policy rule or
    the value function (interpolated linearly) at each point in grid.

    Parameters
    ----------

    w : callable value function (probably instance of LinInterp from last iter)
    u_fn : The period utility function to be maximized. Omega in DH2013
    grid : Domain of w.  This is the real wage today at start of today.
    lambda : float. Degree of wage rigidity. 0 = flexible, 1 = fully rigid
    shock : array. Draws from a lognormal distribution.
    pi : steady-state (for now) inflation level.  Will be changed.

    Returns
    -------

    Tv : The next iteration of the value function v. Instance of LinInterp.
    """
    if grid is None:
        grid = np.linspace(0.1, 4, 100)
    if shock is None:
        sigma = 0.2
        mu = -(sigma ** 2) / 2
        trunc = truncate_normal(norm(loc=mu, scale=sigma), .05, .95)
        shock = np.sort(np.exp(trunc.rvs(30)))

    #--------------------------------------------------------------------------
    vals = []
    w_max = grid[-1]

    h_ = lambda x, ashock: -1 * ((u_fn(x, shock=ashock)) +
                                 beta * w((x / (1 + pi))))

    for y in grid:
        for z in shock:
            m1 = maximizer(h_, 0, w_max, args=(z,))  # can be pre-cached/z
            m2 = maximizer(h_, y, w_max, args=(z,))
            ## THE CONVEX COMBO SHOULD BE OF THE UTILITIES.
            value = -1 * ((1 - lambda_) * h_(m1, z) + lambda_ * h_(m2, z))
            vals.append([y, z, value, m1, m2])
    vals = np.array(vals)
    # split is grid x shocks x [w, z, w*, m1, m2]
    split = np.array(np.split(vals, len(grid)))
    SHOCKS = 1
    FREE = 3
    Tv = LinInterp(grid, split.mean(SHOCKS)[:, 2])  # operate on this
    # Wage(shock).  Doesn't matter which row for free case.
    wage_schedule = LinInterp(shock, split[0][:, FREE])
    return Tv, wage_schedule, vals


def cycle(vs, max_cycles=100):
    """
    # Example
    subshocks = shocks[[0, 250, 500, 750, -1]]

    vs = [(w0, {'shock':subshocks[0]}),
          (w0, {'shock':subshocks[1]}),
          (w0, {'shock':subshocks[2]}),
          (w0, {'shock':subshcks[3]}),
          (w0, {'shock':subshocks[4]})]
    gen = cycle(vs)
    next(gen)
    plt.legend()
    next(gen)
    """
    n_vfs = len(vs)
    try:
        colors = ['k', 'r', 'b', 'g', 'c', 'm', 'y'][:n_vfs]
        colors = it.cycle(colors)
    except IndexError:
        raise('Too many value functions.  Only supports 7.')
    for i in range(max_cycles):
        out = []
        for v, kwargs in vs:
            v = bellman(v, u_fn, **kwargs)
            # import ipdb; ipdb.set_trace()
            # very much hackish on the labeling.
            ax = v.plot(c=next(colors),
                        label='{0}:{1:.4f}'.format(*kwargs.iteritems().next()))
            out.append((v, kwargs))
        vs = out
        yield out, ax


def burn_in_vf(w, maxiter=15, lamda_=.8, pi=2, shock=1, argmax=False):
    grid = w.X
    w_max = grid[-1]

    for i in range(1, maxiter):
        vals = []
        print("{} / {}".format(i, maxiter))
        h_ = lambda x: -1 * ((u_(x, shock=1)) + beta * w((x / (1 + pi))))

        for y in grid:
            m1 = maximizer(h_, 0, w_max)  # can be pre-cached/z
            m2 = maximizer(h_, y, w_max)
            value = -1 * ((1 - lambda_) * h_(m1) + lambda_ * h_(m2))
            if argmax:
                vals.append((m1, m2))
            else:
                vals.append(value)

        vals = np.array(vals)
        if argmax:
            raise NotImplementedError
        else:
            w = LinInterp(grid, vals)
    return w


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
    # This one is just utility.
    h_ = lambda x, shock: -1 * u_(x, shock=shock)
    ws = np.array([fminbound(h_, 0, 3, args=(x,)) for x in shocks])
    us = [u_(x, shock=y) for x, y in zip(ws, shocks)]
    ax = plt.plot(shocks, ws, label='hours', )
    ax = plt.plot(shocks, us, label='utils')
    plt.legend()
    return ax


def unrestricted_wage_shock_schedule(w0, u_fn, shock=None, pi=pi, grid=None):
    if grid is None:
        grid = np.linspace(0.1, 4, 100)
    if shock is None:
        sigma = 0.2
        mu = -(sigma ** 2) / 2
        trunc = truncate_normal(norm(loc=mu, scale=sigma), .05, .95)
        shock = np.sort(np.exp(trunc.rvs(30)))

    w = get_iters_takewhile(w0, tol=.1, maxiter=15, shock=shocks)
    h_ = lambda x, ashock: -1 * (np.mean(u_fn(x, shock=ashock)) +
                                 beta * w((x / (1 + pi))))
    w_max = grid[-1]
    by_shock = []
    for z in shock:
        m1 = maximizer(h_, 0, w_max, args=(z,))  # can be pre-cached/z
        by_shock.append(m1)
    wage_schedule = LinInterp(shocks, np.array(by_shock))
    wage_schedule.plot()
    return wage_schedule


def restricted_wage_shock_schedule(w0, u_fn, shock=None, lambda_=.8, pi=pi, grid=None):
    if grid is None:
        grid = np.linspace(0.1, 4, 100)
    if shock is None:
        sigma = 0.2
        mu = -(sigma ** 2) / 2
        trunc = truncate_normal(norm(loc=mu, scale=sigma), .05, .95)
        shock = np.sort(np.exp(trunc.rvs(30)))

    w = get_iters_takewhile(w0, tol=.1, maxiter=15, shock=shocks)
    h_ = lambda x, ashock: -1 * (np.mean(u_fn(x, shock=ashock)) +
                                 beta * w((x / (1 + pi))))
    w_max = grid[-1]
    by_shock = []
    for y in grid:
        for z in shock:
            m2 = maximizer(h_, y, w_max, args=(z,))
            by_shock.append((y, m2))

    by_shock = np.array(by_shock)
    split = np.split(by_shock, len(shock))
    return split




if __name__ == '__main__':
    pass
