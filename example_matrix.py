import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

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
    # nC = 60
    M = rd.rand(n, n)
    M = np.dot(M, M.transpose())
    M_obj = Quadratic(M)
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
    e = 1
    r = 2
    niter = 10
    x0 = np.ones(n)

    dist_to_solution = lambda x: np.linalg.norm(x - x_opt)
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
        """TODO: to be defined1. """
        self.A = A
        (n, m) = A.shape
        if n != m:
            raise NameError("matrix should be square.")
        self.dim = n

    def __call__(self, x):
        """Funciton evaluation."""
        return 0.5*np.inner(self.A.dot(x), x)

    def prox(self, x, r):
        """Proximal operator.
        Return solution to
        argmin_u <u,Au> + r/2 || u - x||^2
        """
        solution = solve_qp(self.A + r*np.eye(self.dim, self.dim), r*x)
        return solution[0]


if __name__ == "__main__":
    main()
