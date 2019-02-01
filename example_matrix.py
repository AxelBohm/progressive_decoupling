import numpy as np
import numpy.random as rd
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.linalg import eigh

from quadprog import solve_qp
from decoupling import decomposable_decoupling
from util import CallBack

import pdb


def main():
    """TODO: solve min_x <x,Mx>
    by splitting M into A, C, and B
    M = (A   B)
        (B^t C)
    """
    n = 100
    nA = 4
    M = rd.rand(n, n)
    M = np.dot(M, M.T)
    M_obj = Quadratic(M + np.eye(n))

    alpha = M_obj.smallest_eigenvalue()
    P = np.ones((n, n))*1/n
    P_perp = np.eye(n) - P
    gamma = la.norm(la.multi_dot([P_perp, M, P_perp]), 2)
    beta = 0.5*la.norm(la.multi_dot([P, M+M.T, P_perp]), 2)

    e_0 = beta**2/alpha + gamma

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
    plt.title(f'Decoupling, r={r}, e={e}')
    plt.show()

    plt.plot(np.arange(niter), callback.stored_dist_to_solution)
    plt.xlabel('iterations')
    plt.ylabel('distance to solution')
    plt.title(f'Decoupling, r={r}, e={e}')
    plt.show()


class Quadratic():
    """Matrix representing the quadratic function
    x mapsto 1/2 * <x,Ax>
    """

    def __init__(self, A):
        self.A = A
        (n, m) = A.shape
        if n != m:
            raise ValueError("matrix should be square.")
        self.dim = n

    def __call__(self, x):
        """Funciton evaluation."""
        return 0.5*np.inner(self.A.dot(x), x)

    def smallest_eigenvalue(self):
        smallest_eigenvalue, _ = eigh(self.A, eigvals=(0, 0))
        return smallest_eigenvalue

    def largest_eigenvalue(self):
        largest_eigenvalue, _ = eigh(self.A, eigvals=(n-1, n-1))
        return largest_eigenvalue

    def prox(self, x, r):
        """Proximal operator.
        Return solution to
        argmin_u <u,Au> + r/2 || u - x||^2
        """
        solution = solve_qp(self.A + r*np.eye(self.dim, self.dim), r*x)
        return solution[0]


if __name__ == "__main__":
    main()
