cyDPP
====

This repo is a python / cython port of part of Alex Kulesza's Matlab code for sampling from a determinental point process. (http://www.alexkulesza.com/)

It currently only implements sampling from a standard dpp, not a dual or conditional dpp.

Note that I only implemented the cython version as a way of teaching myself cython; it should be slightly faster than the numpy version (especially for sampling a large number of points), although one could likely also make the numpy code faster without using cython.

#### Requirements

- python3
- numpy
- scipy
- matplolib
- (cython)


#### Usage

To run the demo, just run `python demo_python.py` or `python demo_cython.py`, Before running the cython demo, it is necessary to compile the cython code (see below). Use `-h` to see more options.

#### Compilation

To compile the cython code, run `python setup.py build_ext --inplace`


#### Other implementations

Please also see these other implementations:

- https://github.com/javiergonzalezh/dpp
- https://github.com/metalgeekcz/DPP
- https://github.com/chappers/dpp

