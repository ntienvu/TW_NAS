import numpy as np
from cyDPP.sample_k import sample_k


def sample_dpp(vals, vecs, k=None):
    """
    Sample a set of points from a decomposed kernel
    :param vals: sorted eigenvalues
    :param vecs: matrix of corresponding eigenvectors (as columns)
    :param k: number of points to sample (will be chosen randomly if k is None)
    :return: array of indices of samples
    """

    # randomly select k if not given
    if k is None:
        D = vals / (vals+1)
        v = np.nonzero(np.random.rand(len(D)) <= D)[0]
    else:        
        v = sample_k(vals, k)

    k = len(v)
    V = vecs[:, v]
    Y = np.zeros((k, ), dtype=int)

    np.random.seed(14)
    rands = np.random.rand(k)

    for i in range(k, 0, -1):
        # compute probabilities for each item
        P = np.sum(V**2, axis=1)
        P = P / np.sum(P)

        # choose a new item to include
        cumsum = np.cumsum(P)
        random_target = rands[i-1]
        nonzero_indices = np.nonzero(random_target <= cumsum)[0]
        index = nonzero_indices[0]
        Y[i-1] = index

        # choose a vector to eliminate
        nonzero_indices = np.nonzero(V[Y[i-1], :])[0]
        j = nonzero_indices[0]
        Vj = V[:, j]
        n_rows, n_cols = V.shape
        indices = np.array(np.r_[range(1, j+1), range(j+2, n_cols+1)], dtype=int) - 1
        V = V[:, indices]

        diff = np.outer(Vj, V[Y[i-1], :] / Vj[Y[i-1]])
        V = V - diff

        # orthogonalize
        for a in range(0, i-1):
            for b in range(0, a):
                V[:, a] = V[:, a] - np.dot(V[:, a], V[:, b])*V[:, b]
            norm = np.linalg.norm(V[:, a])
            assert norm > 0
            V[:, a] = V[:, a] / norm

    return Y
