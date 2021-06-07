from distutils.core import setup
from Cython.Build import cythonize
setup(ext_modules=cythonize('sample_dpp_cython.pyx'))
