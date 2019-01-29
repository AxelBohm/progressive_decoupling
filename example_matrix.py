import numpy as np
import numpy.random as rd
from quadprog import solve_qp
from decoupling import decomposable_decoupling


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
    # A = M[0:39, 0:39]
    A = quadratic(M[0:nA, 0:nA])
    # C = M[40:99, 40:99]
    C = quadratic(M[nA:n, nA:n])
    # B = M[0:39, 40:99]
    M[0:nA, 0:nA] = 0
    M[nA:n, nA:n] = 0
    B = quadratic(M)

    # elicitation parameter
    e = 5
    # proximal parameter
    r = 1
    # number of iterations
    niter = 100

    x_opt = decomposable_decoupling([A, C, B], niter, e, r, n)
    print(f"approx sol. = {x_opt}")
    f_opt = M_obj(x_opt)
    print(f"obj fun value at approx solution = {f_opt}")


class quadratic():

    """ Quadratic function. """

    def __init__(self, A):
        """TODO: to be defined1. """
        self.A = A
        (n, m) = A.shape
        if n != m:
            raise NameError("matrix should be square.")
        self.dim = n

    def __call__(self, x):
        """Funciton evaluation."""
        return np.inner(self.A.dot(x), x)

    def prox(self, x, r):
        """Proximal operator.
        Return solution to
        argmin_u <u,Au> + r/2 || u - x||^2
        """
        return solve_qp(self.A + r/2*np.eye(self.dim, self.dim), x)


if __name__ == "__main__":
    main()
