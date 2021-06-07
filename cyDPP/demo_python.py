import os
import sys
from optparse import OptionParser

import numpy as np
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt

from decompose_kernel import decompose_kernel
from sample_dpp import sample_dpp


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('-n', dest='n', default=60,
                      help='Size of grid: default=%default')
    parser.add_option('-k', dest='k', default=None,
                      help='Number of points to sample (None=random): default=%default')
    parser.add_option('--seed', dest='seed', default=None,
                      help='Random seed: default=%default')

    (options, args) = parser.parse_args()

    n = int(options.n)
    k = options.k
    if k is not None:
        k = int(k)
    seed = options.seed
    if seed is not None:
        np.random.seed(int(seed))

    # create a grid of possible points and define a kernel
    sigma = 0.1
    y, x = np.mgrid[1:n+1, 1:n+1] / float(n)
    L = np.exp(-((x.T.reshape(n**2, 1) - x.T.reshape(n**2, 1).T)**2 + (y.T.reshape(n**2, 1) - y.T.reshape(n**2, 1).T)**2) / sigma**2)
    
    # decompose it into eigenvalues and eigenvectors
    vals, vecs = decompose_kernel(L)

    # sample points from a DPP
    dpp_sample = sample_dpp(vals, vecs, k=k)

    # also take a purely random sample
    ind_sample = np.random.choice(n*n, size=len(dpp_sample), replace=False)

    # plot the samples 
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ax1, ax2 = axes
    x_vec = x.T.reshape((n**2, ))
    y_vec = y.T.reshape((n**2, ))
    ax1.scatter(x_vec[dpp_sample], y_vec[dpp_sample])
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax2.scatter(x_vec[ind_sample], y_vec[ind_sample])
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax1.set_title('DPP')
    ax2.set_title('Uniform')
    plt.show()


if __name__ == '__main__':
    main()
