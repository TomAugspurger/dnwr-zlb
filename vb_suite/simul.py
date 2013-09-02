from vbench.benchmark import Benchmark

setup = """
import ..model.value_function as vf
from ..model.helpers import load_params
from ..model.gen_interp import Interp

params = load_params()
grid = params['grid'][0]
try:
    w_grid = params['grid'][0]
except KeyError:
    w_grid = params['w_grid'][0]
w0 = Interp(w_grid, -w_grid + 28)
"""

single_bellman_iteration = \
    Benchmark("vf.bellman(w0, params)", setup,
              name='single_bellman_iteration')
