"""
A clean implementation of the value funciton.

v(w) = (1 - lambda_) * (u(today | w' >= 0) + beta * v(w')) +
            lambda_  * (u(today | w' >= w) + beta * v(w'))

"""
from __future__ import division

import numpy as np
import pandas as pd
from scipy.interpolate import pchip

from gen_interp import Interp
from helpers import maximizer
from cfminbound import opt_loop
#-----------------------------------------------------------------------------
np.random.seed(42)


def u_(wage, shock=1, eta=2.5, gamma=0.5, aggL=0.85049063822172699):
    utility = (wage ** (1 - eta) -
              ((gamma / (gamma + 1)) * shock *
              (wage ** (-eta) * aggL) ** ((gamma + 1) / gamma)))
    return utility

#-----------------------------------------------------------------------------


def bellman(w, params, u_fn=u_, lambda_=None, shock=None, pi=None,
            kind=None, grid=None):
    """
    Differs from bellman by optimizing for *each* shock, rather than
    for the mean.  I think this is right since the agent observes Z_{it}
    *before* choosing w_{it}.

    Operate on the bellman equation. Returns the value function
    (interpolated linearly) at each point in grid.

    Parameters
    ----------

    w : callable value function (probably instance of LinInterp from last iter)
    u_fn : The period utility function to be maximized. Omega in DH2013
    grid : Domain of w.  This is the real wage today at start of today.
    lambda : float. Degree of wage rigidity. 0 = flexible, 1 = fully rigid
    shock : array. Draws from a lognormal distribution.
    pi : steady-state (for now) inflation level.  Will be changed.
    kind : type of interpolation.  Defualt taken from w.
        Overridden if not None. str or int.  See scipy.interpolate.interp1d

    Returns
    -------

    Tv : The next iteration of the value function v. Instance of LinInterp.
    wage_schedule : LinInterp. Wage as function of shock.
    vals : everything else. temporary. [(wage, shock, free_w*, res_w*)]
        This will be grid X shocks X 5
    """

    lambda_ = lambda_ or params['lambda_'][0]
    pi = pi or params['pi'][0]

    # Need if since it's checking using or checks truth of array so any/all
    if grid is None:
        grid = params['grid'][0]

    if shock is None:
        shock = params['shock'][0]

    kind = kind or w.kind
    #--------------------------------------------------------------------------
    vals = np.zeros((len(grid), len(shock), 5))
    vals = opt_loop(vals, grid, shock, w, pi, lambda_)

    SHOCKS = 1
    FREE = 3
    Tv = Interp(grid, vals.mean(SHOCKS)[:, 2], kind=kind)  # operate on this
    # Wage(shock).  Doesn't matter which row for free case.
    wage_schedule = Interp(shock, vals[0][:, FREE], kind=kind)
    vals = pd.Panel(vals, items=grid, major_axis=shock,
                    minor_axis=['wage', 'shock', 'value', 'm1', 'm2'])
    vals.major_axis.name = 'shock'

    return Tv, wage_schedule, vals


def g_p(g, f_dist, params, tol=1e-3, full_output=False):
    """
    Once you have the wage/shock schedule, use this to get the distribution
    of wages.

    Parameters
    ----------

    g : instance of pchip.  e.g. pchip(grid, grid/4) with Y
        going from [0, 1].
    f_dist : instance of lognormal with cdf callable.
    tol : tolerance for convergence.

    Returns
    -------

    gp : instance of pchip.  Approximation to wage distribution.
    """
    lambda_ = params['lambda_'][0]
    grid = g.X

    e = 1
    vals = []
    while e > tol:
        gp = Interp(grid, ((1 - lambda_) * f_dist.cdf(grid) +
                    lambda_ * f_dist.cdf(grid) * g.Y), kind='pchip')
        e = np.max(np.abs(gp.Y - g.Y))
        print("The error is {}".format(e))
        g = gp
        if full_output:
            vals.append(g)
    if full_output:
        return gp, vals
    else:
        return gp


def get_rigid_output(params, ws, flex_ws, gp):
    """
    Eq 18 in DH. Don't actually use this. p3 is slow.

    Parameters
    ----------

    params : dict of parameters
    ws : rigid wage schedule.  Callable e.g. instance of LinInterp.
    flex_ws: flexible wage schedule.  Also callable.
    gp: Probability densitity function for wage distribution.
        Derived from g_p.

    Returns
    -------

    output: float.  Also equal to labor in this model.
    """
    sigma, grid, shock, eta, gamma, pi = (params['sigma'][0], params['grid'][0],
                                          params['shock'][0], params['eta'][0],
                                          params['gamma'][0], params['pi'][0])
    lambda_ = params['lambda_'][0]
    sub_w = lambda z: grid[grid > ws(z)]  # TODO: check on > vs >=
    dg = pchip(gp.X, gp.Y).derivative

    p1 = ((1 / shock) ** (gamma * (eta - 1) / (gamma + eta)) *
          (flex_ws(shock) / ws(shock)) ** (eta - 1)).mean()

    p2 = ((1 / shock) ** (gamma * (eta - 1) / (gamma + eta)) *
          gp(ws(shock) * (1 + pi)) * (flex_ws(shock) / ws(shock)) ** (eta - 1)).mean()

    inner_f = lambda w, z: ((1 + pi) * dg(w * (1 + pi)) *
                            (flex_ws(z) / w)**(eta - 1))

    p3 = 0.0
    for z in shock:
        inner_range = sub_w(z)
        inner_vals = inner_f(inner_range, z).mean()
        p3 += (1 / z)**(gamma * (eta - 1) / (eta + gamma)) * inner_vals

    p3 = p3 / len(shock)

    # z_part is \tilde{Z} in my notes.
    z_part = ((1 - lambda_) * p1 +
              lambda_ * (p2 + p3))**(-(eta + gamma) / (gamma * (eta - 1)))

    return ((eta - 1) / eta)**(gamma / (1 + gamma)) * (1 / z_part)**(gamma / (1 + gamma))


def burn_in_vf(w, params, maxiter=15, shock=1, kind=None):
    """
    Use to get a rough shape of the vf.

    Parameters
    ----------

    w : Interp :: value function.
    params: dict :: parameters
    maxiter: int :: number of times to run
    shock: float or array-like
    kind: str :: interpolation kind
    Returns
    -------

    burned :: Interp
    """
    lambda_, pi, beta = params['lambda_'][0], params['pi'][0], params['beta'][0]
    try:
        grid = w.X
    except AttributeError:
        grid = grid

    w_max = grid[-1]
    kind = kind or w.kind

    for i in range(1, maxiter + 1):
        vals = []
        print("{} / {}".format(i, maxiter))
        h_ = lambda x: -1 * ((u_(x, shock=1)) + beta * w((x / (1 + pi))))

        for y in grid:
            m1 = maximizer(h_, 0, w_max)  # can be pre-cached/z
            m2 = maximizer(h_, y, w_max)
            value = -1 * ((1 - lambda_) * h_(m1) + lambda_ * h_(m2))
            vals.append(value)

        vals = np.array(vals)
        w = Interp(grid, vals, kind=kind)
    return w


def iter_bellman(v, tol=1e-3, maxiter=100, strict=True, log=True, **kwargs):
    """
    """
    params = kwargs.pop('params')
    e = 1
    vfs, wss, rests, es = [], [], [], []
    for i in range(maxiter):
        Tv, ws, rest = bellman(v, params, **kwargs)
        e = np.max(np.abs(Tv.Y - v.Y))
        print("At iteration {} the error is {}".format(i, e))
        if e < tol:
            if log:
                return Tv, ws, rest, vfs, wss, rests, ws
            else:
                return Tv, ws, rest
        if log:
            vfs.append(Tv)
            wss.append(ws)
            rests.append(rest)
            es.append(es)

        v = Tv
    else:
        print("Returning before convergence! specified tolerance was {},"
              " but current error is {}".format(tol, e))
        if strict:
            raise ValueError
        else:
            return Tv, ws, rest
