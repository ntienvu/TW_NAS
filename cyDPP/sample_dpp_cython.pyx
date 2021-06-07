import sys
import numpy as np
from libc.math cimport sqrt
import sample_k


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
        v = sample_k.sample_k(vals, k)


    Y = do_sampling(int(len(v)), int(len(vals)), np.array(v, dtype=np.int32), np.array(vecs, dtype=float))
    # convert Y back to a numpy array
    Y = np.array(Y, dtype=np.int32)
    return Y


cdef do_sampling(int k, int n, int[:] v, double[:, :] vecs):

    cdef int a, b, c, i
    cdef int ncols = k
    cdef double[:, ::1] V = np.empty([n, k], dtype=np.double)
    cdef double dp
    cdef double norm
    cdef double[:] P = np.empty((n, ), dtype=np.double)
    cdef double[:] P_cumsum = np.empty((n, ), dtype=np.double)
    cdef double[:] Vj = np.empty((n, ), dtype=np.double)
    cdef double[:, ::1] diff = np.empty([n, n], dtype=np.double)
    cdef double P_sum
    cdef int index
    cdef int[:] Y = np.empty(k, dtype=np.int32)

    a = 0 
    while a < n:
        b = 0
        while b < k:
            V[a, b] = vecs[a, v[b]]
            b += 1
        a += 1

    # generate the random numbers that we will need
    rands = np.random.rand(k)

    i = k
    while i > 0:
        # compute probabilities for each item
        P_sum = 0.0
        a = 0
        while a < n:
            P[a] = V[a, 0] * V[a, 0]
            b = 1
            while b < ncols:
                P[a] = P[a] + V[a, b] * V[a, b]
                b += 1
            P_sum += P[a]
            a += 1

        a = 0
        while a < n:
            P[a] = P[a] / P_sum
            P_cumsum[a] = P[a]
            if a > 0:
                P_cumsum[a] += P_cumsum[a-1]
            if rands[i-1] <= P_cumsum[a]:
                index = a
                a = n
            a += 1

        Y[i-1] = index

        b = 0
        while b < ncols:
            if V[Y[i-1], b] != 0:
                j = b
                b = ncols
            b += 1

        a = 0
        while a < n:
            Vj[a] = V[a, j]
            a += 1
        
        a = 0
        while a < n:
            b = j
            while b < ncols-1:
                V[a, b] = V[a, b+1]
                b += 1
            a += 1

        a = 0
        while a < n:
            b = 0
            while b < ncols:
                diff[a, b] = Vj[a] * V[Y[i-1], b] / Vj[Y[i-1]]
                b += 1
            a += 1

        a = 0
        while a < n:
            b = 0
            while b < ncols:
                V[a, b] = V[a, b] - diff[a, b]       
                b += 1
            a += 1

        ncols = ncols - 1

        a = 0
        while a < i-1:
            b = 0
            while b < a:
                dp = 0.0
                c = 0
                while c < n:
                    # np.dot(V[:, a], V[:, b])
                    dp += V[c, a] * V[c, b]
                    c += 1
                c = 0
                while c < n:
                    #V[:, a] = V[:, a] - np.dot(V[:, a], V[:, b])*V[:, b]
                    V[c, a] = V[c, a] - dp * V[c, b]
                    c += 1
                b += 1
            #norm = np.linalg.norm(V[:, a])
            c = 0
            norm = 0.0
            while c < n:
                norm += V[c, a] * V[c, a]
                c += 1
            norm = sqrt(norm)

            c = 0
            while c < n:
                V[c, a] = V[c, a] / norm
                c += 1
            a += 1        

        i = i - 1

    return Y