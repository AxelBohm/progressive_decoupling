import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

from decoupling import decomposable_decoupling
from util import CallBack, Quadratic, random_psd, projection_onto_linkage_space

import pdb


def main():
    """TODO: solve min_x <x,Mx>
    by splitting M into A, C, and B
    M = (A   B)
        (B^t C)
    """
    n = 100
    nA = 4

    # construct PSD matrix
    M = random_psd(n)
    M_obj = Quadratic(M + np.eye(n))

    # construct split
    A = M.copy()
    A[nA:n, :] = 0
    A[:, nA:n] = 0
    A = Quadratic(A)
    C = M.copy()
    C[0:nA, :] = 0
    C[:, 0:nA] = 0
    C = Quadratic(C)
    B = M.copy()
    B[0:nA, 0:nA] = 0
    B[nA:n, nA:n] = 0
    B = Quadratic(M)

    # construct projections
    P = projection_onto_linkage_space(n, 3)
    P_perp = np.eye(3*n) - P

    # construct objective function in product space
    M_large = np.zeros((3*n, 3*n))
    M_large[:n, :n] = A.matrix
    M_large[n:2*n, n:2*n] = B.matrix
    M_large[2*n:, 2*n:] = C.matrix

    alpha = M_obj.smallest_eigenvalue()
    gamma = la.norm(la.multi_dot([P_perp, M_large, P_perp]), 2)
    beta = 0.5*la.norm(la.multi_dot([P, M_large+M_large.T, P_perp]), 2)

    e_0 = beta**2/alpha + gamma

    # global minimizer
    x_opt = np.zeros(n)

    # params
    e = np.round(e_0)
    r = e + 100
    # r = np.sqrt((e/2)**2+1) + e/2
    niter = 100
    x0 = np.ones(n)

    def dist_to_solution(x): return la.norm(x - x_opt)
    callback = CallBack(M_obj, dist_to_solution)

    x_opt = decomposable_decoupling([A, C, B], niter, e, r, n, x0=x0,
                                    callback=callback)

    # make plots
    plt.plot(np.arange(niter), callback.stored_obj_fun_values)
    plt.xlabel('iterations')
    plt.ylabel('function value')
    plt.title(f'Decoupling, Matrix, indefinite, r={r}, e={e}')
    plt.show()

    plt.plot(np.arange(niter), callback.stored_dist_to_solution)
    plt.xlabel('iterations')
    plt.ylabel('distance to solution')
    plt.title(f'Decoupling, Matrix, indefinite, r={r}, e={e}')
    plt.show()


if __name__ == "__main__":
    main()
