import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

from decoupling import decomposable_decoupling
from util import CallBack, Quadratic, random_psd, projection_onto_linkage_space

import pdb


def main():
    """TODO: solve min_x <x,Mx>
    by splitting M into A, and C
    M = (A 0)
        (0 C)
    """
    n = 100

    # construct SPD matrix
    A = Quadratic(random_psd(n))
    C = Quadratic(random_psd(n))
    M = A + C

    # construct projections
    P = projection_onto_linkage_space(n, 2)
    P_perp = np.eye(2*n) - P

    # construct objective function in product space
    M_large = block_diag(A.matrix, C.matrix)

    alpha = M.smallest_eigenvalue()
    gamma = la.norm(la.multi_dot([P_perp, M_large, P_perp]), 2)
    beta = 0.5*la.norm(la.multi_dot([P, M_large+M_large.T, P_perp]), 2)

    e_0 = beta**2/alpha + gamma

    # global minimizer
    x_opt = np.zeros(n)

    # params
    # e = np.round(e_0)
    # r = e + 100
    # r = np.sqrt((e/2)**2+1) + e/2
    e = 0.01
    r = 0.02
    niter = 10
    x0 = np.ones(n)

    def dist_to_solution(x): return la.norm(x - x_opt)
    callback = CallBack(M, dist_to_solution)

    x_opt = decomposable_decoupling([A, C], niter, e, r, n, x0=x0,
                                    callback=callback)

    # make plots
    plt.plot(np.arange(niter), callback.stored_obj_fun_values)
    plt.xlabel('iterations')
    plt.ylabel('function value')
    plt.title(f'Decoupling, Matrix, convex, r={r}, e={e}')
    plt.show()

    plt.plot(np.arange(niter), callback.stored_dist_to_solution)
    plt.xlabel('iterations')
    plt.ylabel('distance to solution')
    plt.title(f'Decoupling, Matrix, convex, r={r}, e={e}')
    plt.show()


if __name__ == "__main__":
    main()
