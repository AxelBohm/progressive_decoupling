import numpy as np
import numpy.random as rd
from quadprog import solve_qp
from decoupling import decomposable_decoupling
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
    M_obj = quadratic(M)
    A = M.copy()
    A[nA:n, :] = 0
    A[:, nA:n] = 0
    A = quadratic(A)
    C = M.copy()
    C[0:nA, :] = 0
    C[:, 0:nA] = 0
    C = quadratic(C)
    B = M.copy()
    B[0:nA, 0:nA] = 0
    B[nA:n, nA:n] = 0
    B = quadratic(M)

    # elicitation parameter
    e = 5
    # proximal parameter
    r = 1
    # number of iterations
    niter = 100
    # initial value
    x0 = np.ones(n)

    x_opt, fun_vals = decomposable_decoupling([A, C, B], niter, e, r, n, x0=x0)
    print(f"approx sol. = {x_opt}")
    f_opt = M_obj(x_opt)
    print(f"obj fun value at approx solution = {f_opt}")


class quadratic():

    """ Matrix representing the quadratic function
    x \mapsto 1/2 * <x,Ax>
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
