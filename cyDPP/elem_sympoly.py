import numpy as np

def elem_sympoly(eigenvals, k):
    """
    Given a vector of values and a maximum size k, determine the value of
    the elementary symmetric polynomials:
    E(l+1,n+1) = sum_{J \subseteq 1..n,|J| = l} prod_{i \in J} lambda(i)
    """
    N = len(eigenvals)
    E = np.zeros([k+1, N+1])
    E[0, :] = 1
    for l in range(1, k+1):
        for n in range(1, N+1):
            E[l, n] = E[l, n-1] + eigenvals[n-1] * E[l-1, n-1]
    return E
