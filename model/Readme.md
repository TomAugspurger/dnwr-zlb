Model
-----

This section contains the code that simulates the model.
It is divided into several sections:

To run the simulation, simply open up a terminal and type:

```bash
python run_value_function.py parameters.json
```

### run_value_function.py

If you've never read a Python source file, one thing to look for is the

```python

if __name__ == '__main__':
    main()
```

located at the end of many Python scripts.  When a script is called from
the command line, as above, this is the code that gets executed.

This file serves two purposes.  First, it is a think wrapper around
the real work, mostly contained in `value_function.py`.
Second, it handles the parallelization via the `joblib` module.
This is an embarrassingly parallel problem over the space of pi and lambda,
in my simplistic implementation they are run completely independently.
A more sophisticated program could probably be written, but this is a
first try.

Once the value function has stabilized (see the next section), we can
sample from the implied steady state values to get an empirical cumulative
distribution function for the wages. This is done using a slightly modified
version of John Stachurski's `ecdf.py` module.

We use that ECDF to get an estimate of the density using Scipy's `kde` module.
This is one section where I deviate from Daly and Hobijn's approach.
They used piecewise cubic polynomials to approximate the wage distribution.
I started down this path, but had better luck with the ECDF and KDE approach.

With an estimated density in hand we go on the approximate the equation
for aggregate output in the rigid case.  Here we make good
use of Scipy's `FITPACK` wrapper for the numerical integration.

### value_function.py

As mentioned above, the bulk of the work is done here.
The central function is `bellman`, which takes a wage and some other parameters.
There's some setup before getting to the line

```python
vals = opt_loop(vals, w_grid, z_grid, w, pi, lambda_, aggL)
```

This is the central optimization loop, and so I've written it in Cython,
which is compiled down to C.  I've also included a Python version in `py_opt_loop`,
which contains the same logic but is much slower.

The rest of the function should be a straightforward implementation of the
value function from the paper.  We get the wage schedule from the unconstrained workers.
