import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

import decoupling as dc
import util as ut

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
    M = ut.random_psd(n)

    # construct split
    A = M.matrix.copy()
    A[nA:n, :] = 0
    A[:, nA:n] = 0
    A = ut.Quadratic(A)
    C = M.matrix.copy()
    C[0:nA, :] = 0
    C[:, 0:nA] = 0
    C = ut.Quadratic(C)
    B = M.matrix.copy()
    B[0:nA, 0:nA] = 0
    B[nA:n, nA:n] = 0
    B = ut.Quadratic(B)

    # construct projections
    P = ut.projection_onto_linkage_space(n, 3)

    # construct objective function in product space
    M_large = np.zeros((3*n, 3*n))
    M_large[:n, :n] = A.matrix
    M_large[n:2*n, n:2*n] = B.matrix
    M_large[2*n:, 2*n:] = C.matrix
    M_large = block_diag(A.matrix, B.matrix, C.matrix)
    M_large = ut.Quadratic(M_large)

    e_0 = ut.compute_minimal_elicitation(M, M_large, P)
    pdb.set_trace()
    # global minimizer
    x_opt = np.zeros(n)

    # params
    e = np.round(e_0)
    r = e + 100
    # r = np.sqrt((e/2)**2+1) + e/2
    niter = 100
    x0 = np.ones(n)

    def dist_to_solution(x): return la.norm(x - x_opt)
    callback = ut.CallBack(M, dist_to_solution)

    x_opt = dc.decomposable_decoupling([A, C, B], niter, e, r, n, x0=x0,
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
