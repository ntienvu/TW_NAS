import numpy as np
from scipy.linalg import eigh

def decompose_kernel(M):
    """
    Decompose a symmetric matrix into sorted eigenvalues and corresponding eigenvectors
    :param M: input square np.array
    :return vals, vecs: vector of sorted eigenvalues, matrix of corresponding eigenvectors
    """
    vals, vecs = np.linalg.eigh(M)
    vals = np.real(vals)
    vecs = np.real(vecs)
    order = np.argsort(vals)
    return vals[order], vecs[:, order]
