from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("cfminbound", ["cfminbound.pyx"]),
                 Extension("gen_interp", ["gen_interp.pyx"])],
    include_dirs=[np.get_include()]
)
