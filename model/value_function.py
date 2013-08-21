"""
A clean implementation of the value funciton.

v(w) = (1 - lambda_) * (u(today | w' >= 0) + beta * v(w')) +
            lambda_  * (u(today | w' >= w) + beta * v(w'))

"""
from __future__ import division

import numpy as np
import pandas as pd
from scipy.interpolate import pchip
from scipy.integrate import quad
import statsmodels.api as sm

from gen_interp import Interp
from helpers import maximizer, truncated_draw, sample_path
from cfminbound import opt_loop
from ecdf import ecdf
#-----------------------------------------------------------------------------
np.random.seed(42)


def u_(wage, shock=1, eta=2.5, gamma=0.5, aggL=0.85049063822172699):
    utility = (wage ** (1 - eta) -
              ((gamma / (gamma + 1)) * shock *
              (wage ** (-eta) * aggL) ** ((gamma + 1) / gamma)))
    return utility

#-----------------------------------------------------------------------------


def bellman(w, params, u_fn=u_, lambda_=None, z_grid=None, pi=None,
            kind=None, w_grid=None):
    """
    Differs from bellman by optimizing for *each* shock, rather than
    for the mean.  I think this is right since the agent observes Z_{it}
    *before* choosing w_{it}.

    Operate on the bellman equation. Returns the value function
    (interpolated linearly) at each point in grid.

    Parameters
    ----------

    w : callable value function (probably instance of LinInterp from last iter)
    u_fn : The period utility function to be maximized. Omega in DH`2013
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
    #-------------------------------------------------------------------------
    A note on `shock`:  Why am I taking draws from a distribution?
    I optimize at each draw, and then take the mean over those.  But
    wouldn't it be better to take a uniform sample over the *support*
    of the distribution, and then weight the resulting utilities by
    the *pdf at that point*?  This branch (`rethink_distribution`) attempts
    to implement this alternative strategy.
    """

    lambda_ = lambda_ or params['lambda_'][0]
    pi = pi or params['pi'][0]
    ln_dist = params['full_ln_dist'][0]

    # Need if since it's checking using or checks truth of array so any/all
    if w_grid is None:
        w_grid = params['w_grid'][0]

    if z_grid is None:
        z_grid = params['z_grid'][0]

    kind = kind or w.kind
    #--------------------------------------------------------------------------
    vals = np.zeros((len(w_grid), len(z_grid), 5))
    vals = opt_loop(vals, w_grid, z_grid, w, pi, lambda_)

    vals = pd.Panel(vals, items=w_grid, major_axis=z_grid,
                    minor_axis=['wage', 'z_grid', 'value', 'm1', 'm2'])

    vals.items.name = 'w_grid'
    vals.major_axis.name = 'z_grid'

    weights = pd.Series(ln_dist.pdf(vals.major_axis.values.astype('float64')),
                        index=vals.major_axis)
    weights /= weights.sum()

    weighted_values = vals.apply(lambda x: x * weights).sum(axis='major').ix['value']
    Tv = Interp(w_grid, weighted_values.values, kind=kind)
    # Wage(z_grid).  Doesn't matter which row for free case.
    wage_schedule = Interp(z_grid, vals.iloc[0]['m1'].values, kind=kind)
    return Tv, wage_schedule, vals


def _handle_solo_grid(zs, grid, good_grid):
    """
    Sometimes only one value isn't nan.
    """
    good_pos = np.where(grid == good_grid)[0]
    good_val = grid[good_pos]
    if good_pos in [0, len(grid) - 1]:
        raise IndexError("Cannot deal with endpoints.")

    nlp, nhp = good_pos - 1, good_pos + 1
    DEC = .95
    nlv = DEC * grid[nlp] + (1 - DEC) * good_val
    nhv = DEC * grid[nhp] + (1 - DEC) * good_val
    ok = False
    while not ok:
        if np.isnan(zs(nlv)):
            nlv = DEC * nlv + (1 - DEC) * good_val
        if np.isnan(zs(nhv)):
            nhv = DEC * nhv + (1 - DEC) * good_val
        if not np.isnan([zs(nlv), zs(nhp)]).all():
            ok = True

    return nlv, nhv


def get_rigid_output(ws, params, flex_ws, g, shocks):
    """

    Eq 18 in DH.

    Parameters
    ----------

    ws : rigid wage schedule.  Callable e.g. instance of Interp.
    params : dict of parameters
    flex_ws: flexible wage schedule.  Also callable.
    g: CDF of wages.  Probably instance of ecdf.
    shocks : shocks that generated g.

    Returns
    -------

    output: float.  Also equal to labor in this model.
    """
    sigma, eta, gamma, pi = (params['sigma'][0],
                             params['eta'][0], params['gamma'][0],
                             params['pi'][0])

    shocks = np.sort(shocks)
    dg = sm.nonparametric.KDEUnivariate(g.observations)
    dg.fit()

    lambda_ = params['lambda_'][0]
    w_grid = np.sort(ws(shocks))  # check on this.  It is in a way
    wmax = w_grid[-1]
    sub_w = lambda z: w_grid[w_grid > ws(z)]  # TODO: check on > vs >=

    p1 = ((1 / shocks) ** (gamma * (eta - 1) / (gamma + eta)) *
          (flex_ws(shocks) / ws(shocks)) ** (eta - 1)).mean()

    p2 = ((1 / shocks) ** (gamma * (eta - 1) / (gamma + eta)) *
          g(ws(shocks) * (1 + pi)) *
          (flex_ws(shocks) / ws(shocks)) ** (eta - 1)).mean()

    inner_f = lambda w, z: ((1 + pi) * dg.evaluate(w * (1 + pi)) *
                            (flex_ws(z) / w)**(eta - 1))

    p3 = 0.0
    for z in shocks[:-1]:  # integrate over empty range for very last shock
        inner_range = sub_w(z)
        a = inner_range[0]
        inner_vals = quad(inner_f, a, wmax, args=z)[0]
        p3 += (1 / z)**(gamma * (eta - 1) / (eta + gamma)) * inner_vals

    p3 = p3 / len(shocks)

    # z_part is \tilde{Z} in my notes.
    z_part = ((1 - lambda_) * p1 +
              lambda_ * (p2 + p3))**(-(eta + gamma) / (gamma * (eta - 1)))

    return ((eta - 1) / eta)**(gamma / (1 + gamma)) * (1 / z_part)**(gamma / (1 + gamma))


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


def taylor_rule(y, pi, ybar, pibar, gy, gpi, beta):
    """
    Equation 14 in DH.

    Parameters
    ----------

    y : output at time t
    pi : inflation at time t
    ybar : steady state output
    pibar : target inflation
    gy : reaction coefficent to output gap
    gpi : reaction coefficent to inflation gap
    beta : discount factor

    Returns
    -------

    i : nominal interest rate
    """
    p1 = (1 + pibar) / beta * (y / ybar) ** gy
    p2 = ((1 + pi) / (1 + pibar)) ** (1 + gpi)
    return p1 * p2 - 1


#-----------------------------------------------------------------------------


def py_opt_loop(w_grid, z_grid, w, vals, params):
    """Python dropin for cfminbound.opt_loop"""
    lambda_ = params['lambda_'][0]
    beta = params['beta'][0]
    pi = params['pi'][0]
    vals = np.zeros((len(w_grid), len(z_grid), 5))
    w_max = w_grid[-1]

    # See equatioon 13 in DH
    h_ = lambda x, ashock: -1 * ((u_(x, shock=ashock)) +
                                 beta * w((x / (1 + pi))))

    for i, y in enumerate(w_grid):
        for j, z in enumerate(z_grid):
            if y == w_grid[0]:
                m1 = maximizer(h_, 0, w_max, args=(z,))  # can be pre-cached/z
            else:
                m1 = vals[0, j, 3]  # first page, shock j, m1 in 3rd pos.
            m2 = maximizer(h_, y, w_max, args=(z,))
            value = -1 * ((1 - lambda_) * h_(m1, z) + lambda_ * h_(m2, z))
            vals[i, j] = (y, z, value, m1, m2)
    return vals
