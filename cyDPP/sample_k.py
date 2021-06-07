import numpy as np
from cyDPP.elem_sympoly import elem_sympoly


def sample_k(eigenvals, k):
    """
    Sample a given number of eigenvalues according to p(S) \propto prod eigenvals \in S
    """
    E = elem_sympoly(eigenvals, k)

    i = len(eigenvals)
    remaining = k

    S = np.zeros((k, ), dtype=int)
    while remaining > 0:
        # compute marginal of i given that we choose remaining values from 1:i
        if i == remaining:
            marg = 1
        else:
            marg = eigenvals[i-1] * E[remaining-1, i-1] / E[remaining, i]

        # sample marginal
        rand = np.random.rand()
        if rand < marg:
            S[remaining-1] = i-1
            remaining = remaining - 1
        i = i-1

    return S
