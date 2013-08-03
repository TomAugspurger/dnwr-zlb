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
        trunc = truncate_distribution(norm(loc=mu, scale=sigma), .05, .95)
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


def restricted_wage_shock_schedule(w0, u_fn, shock=None, lambda_=.8, pi=pi,
                                   grid=None):
    if grid is None:
        grid = np.linspace(0.1, 4, 100)
    if shock is None:
        sigma = 0.2
        mu = -(sigma ** 2) / 2
        trunc = truncate_distribution(norm(loc=mu, scale=sigma), .05, .95)
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
