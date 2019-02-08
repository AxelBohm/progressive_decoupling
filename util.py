import numpy as np
import numpy.random as rd
from scipy.linalg import eigh
from quadprog import solve_qp


class CallBack():

    """Compute objective function value and distance to the solution. """

    def __init__(self, fun, dist_to_solution):
        # self.x_opt = x_opt
        self.obj_fun = fun
        self.dist_to_solution = dist_to_solution

        self.stored_iterates = []
        self.stored_obj_fun_values = []
        self.stored_dist_to_solution = []

    def __call__(self, iterate):
        self.stored_iterates.append(iterate)
        self.stored_obj_fun_values.append(self.obj_fun(iterate))
        self.stored_dist_to_solution.append(self.dist_to_solution(iterate))


class Quadratic():
    """Matrix representing the quadratic function
    x mapsto 1/2 * <x,Ax>
    """

    def __init__(self, A):
        (n, m) = A.shape
        if n != m:
            raise ValueError("matrix should be square.")
        if not np.allclose(A, A.T):
            raise ValueError("matrix should be symmetric")
        self.dim = n
        self.matrix = A

    def __call__(self, x):
        """Funciton evaluation."""
        return 0.5*np.inner(self.matrix.dot(x), x)

    def __add__(self, other):
        return Quadratic(self.matrix + other.matrix)

    def smallest_eigenvalue(self):
        smallest_eigenvalue, _ = eigh(self.matrix, eigvals=(0, 0))
        return smallest_eigenvalue

    def largest_eigenvalue(self, n):
        largest_eigenvalue, _ = eigh(self.matrix, eigvals=(self.dim-1,
                                     self.dim-1))
        return largest_eigenvalue

    def prox(self, x, r):
        """Proximal operator.
        Return solution to
        argmin_u <u,Au> + r/2 || u - x||^2
        """
        res = solve_qp(self.matrix + r*np.eye(self.dim, self.dim), r*x)
        return res[0]


def projection_onto_linkage_space(dim, size_product):
    """generate matrix representing the projection onto the linkage space.
    """
    block = np.eye(dim)*1/size_product
    block = np.repeat(block, size_product, axis=0)
    return np.repeat(block, size_product, axis=1)


def random_psd(n):
    """generates a random  matrix
    """
    C = rd.rand(n, n)
    return C.T @ C
