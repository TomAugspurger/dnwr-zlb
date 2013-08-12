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
from helpers import maximizer, truncated_draw
from cfminbound import opt_loop
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


def g_p(g, ws, params, tol=1e-3, full_output=False):
    """
    Once you have the wage/shock schedule, use this to get the distribution
    of wages.

    Parameters
    ----------

    g : instance of pchip.  e.g. pchip(grid, grid/4) with Y
        going from [0, 1].
    tol : tolerance for convergence.

    Returns
    -------

    gp : instance of pchip.  Approximation to wage distribution.
    """
    lambda_ = params['lambda_'][0]
    grid = g.X
    f_dist = params['full_ln_dist'][0]
    pi = params['pi'][0]

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
    # z_t(w) in the paper; zs :: wage -> shock
    # Was having trouble with them choosing wages only on a subset of grid.
    # Then when I invert I try to map z :: w -> shock I got a bunch of NaNs
    # since most of the grid was *NOT* covered by the range of ws = z^-1.
    # I'm renormalizing the grid to cover *just* the area chosen by our
    # guys.  Need to be careful at the edgees... here and in cfminbound.
    zs = ws.inverse()
    good_grid = grid[~np.isnan(zs(grid))]
    if len(good_grid) == 1:
        new_low, new_high = _handle_solo_grid(zs, grid, good_grid)
        new_grid = np.linspace(new_low, new_high, params['wn'][0])
        # g = Interp(new_grid, new_grid/new_grid[-1], kind='pchip')
    else:
        new_low, new_high = good_grid[0], good_grid[-1]
        new_grid = np.linspace(new_low, new_high, params['wn'][0])

    e = 1
    vals = []
    while e > tol:
        gp = Interp(grid, ((1 - lambda_) * f_dist.cdf(zs(new_grid)) +
                    lambda_ * f_dist.cdf(zs(new_grid)) * g.Y * (1 + pi)),
                    kind='pchip')
        e = np.max(np.abs(gp.Y - g.Y))
        print("The error is {}".format(e))
        g = gp
        if full_output:
            vals.append(g)
    if full_output:
        return gp, vals
    else:
        return gp


def get_rigid_output(ws, params, flex_ws, gp, kind='lognorm', size=1000):
    """
    This will need to change when using grid.

    Eq 18 in DH.

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
    sigma, w_grid, eta, gamma, pi = (params['sigma'][0], params['w_grid'][0],
                                     params['eta'][0], params['gamma'][0],
                                     params['pi'][0])
    shocks = np.sort(truncated_draw(params, lower=.05, upper=.95,
                                    kind=kind, size=size))
    lambda_ = params['lambda_'][0]
    sub_w = lambda z: w_grid[w_grid > ws(z)]  # TODO: check on > vs >=
    dg = pchip(gp.X, gp.Y).derivative

    p1 = ((1 / shocks) ** (gamma * (eta - 1) / (gamma + eta)) *
          (flex_ws(shocks) / ws(shocks)) ** (eta - 1)).mean()

    p2 = ((1 / shocks) ** (gamma * (eta - 1) / (gamma + eta)) *
          gp(ws(shocks) * (1 + pi)) * (flex_ws(shocks) / ws(shocks)) ** (eta - 1)).mean()

    inner_f = lambda w, z: ((1 + pi) * dg(w * (1 + pi)) *
                            (flex_ws(z) / w)**(eta - 1))

    p3 = 0.0
    for z in shocks:
        inner_range = sub_w(z)
        inner_vals = inner_f(inner_range, z).mean()
        p3 += (1 / z)**(gamma * (eta - 1) / (eta + gamma)) * inner_vals

    p3 = p3 / len(shocks)

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
