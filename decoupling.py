import numpy as np


def decomposable_decoupling(phi, niter, e, r, n, **kwargs):
    """
    The splitting case. This means that the linkage is given by the subspace
    S = span (1,1,1,...).

    :phi: a list of functions
    :returns: approx. solution

    """

    x0 = kwargs.pop('x0', None)
    if x0 is None:
        x0 = np.zeros(n)

    y0 = kwargs.pop('y0', None)
    if y0 is None:
        y0 = np.zeros(n)

    x_hat, x = x0.copy(), x0.copy()
    y = y0.copy()

    numof_summands = len(phi)

    for k in range(niter):
        for j in range(numof_summands):
            x_hat[j] = phi[j].prox((x + 1/r*y[j]), r)
            x = np.average(x_hat)
            y = y - (r - e)*(x_hat - x)

    return x
