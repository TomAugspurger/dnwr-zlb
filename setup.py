from distutils.core import setup

import os
import sys
import shutil
import warnings

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
                'vb_suit'],
      )
