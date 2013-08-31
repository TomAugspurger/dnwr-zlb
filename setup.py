from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import os
import sys
import shutil
import warnings

import numpy as np

# may need to work around setuptools bug by providing a fake Pyrex
try:
    import Cython
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "fake_pyrex"))
except ImportError:
    pass


setup(name='dnwr-zlb',
      version='0.1.0',
      description='Reproduction file for my second year paper.',
      author='Tom Augspurger',
      author_email='thomas-augspurger@uiowa.edu',
      packages=['model',
                'data_wrangling',
                'data_wrangling.cps_wrangling',
                'data_wrangling.cps_wrangling.panel_construction',
                'data_wrangling.greenbook_wrangling',
                'data_wrangling.nipa_wrangling',
                'writing',
                'vb_suite'],
      cmdclass={'build_ext': build_ext},
      ext_modules=[Extension("model/cfminbound", ["model/cfminbound.pyx"])],
      include_dirs=[np.get_include()]
      )
