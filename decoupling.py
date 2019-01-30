import numpy as np
import pdb


def decomposable_decoupling(phi, niter, e, r, n, **kwargs):
    """
    solve min_x \sum_i phi_i(x)
    via splitting. This means that the linkage is given by the subspace
    S = span (1,1,1,...).

    :phi: a list of functions
    :returns: approx. solution

    """
    if r <= 0:
        NameError('stepsize parameter should be positive.')

    x0 = kwargs.pop('x0', None)
    if x0 is None:
        x0 = np.zeros(n)

    numof_summands = len(phi)

    y0 = kwargs.pop('y0', None)
    if y0 is None:
        y0 = np.zeros((numof_summands, n))

    x = x0.copy()
    x_hat = np.tile(x0.copy(), (numof_summands, 1))
    y = y0.copy()
    store_fun_val = [0]*niter

    for k in range(niter):
        for j in range(numof_summands):
            x_hat[j] = phi[j].prox((x + 1/r*y[j]), r)
            # projection onto linkage subspace
            x = np.average(x_hat, axis=0)
            y[j] = y[j] - (r - e)*(x_hat[j] - x)

        store_fun_val[k] = sum([phi_i(x) for phi_i in phi])

    return x, store_fun_val
