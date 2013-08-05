Model
-----

This section contains the code that simulates the model.
It is divided into several sections:

    * `Helpers`: various functions used in other files.
    * `value_vunction`: start here.  Implements the value function iteration.
    * `gen_interp`: Generalization of John Stachurski's small `LinInterp` class.
    * `cfminbound`: Cythonized version of the minimization routine used in `bellman`.
    * `run_value_function` Parallel run of the model.
