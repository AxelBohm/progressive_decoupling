import numpy as np
import numpy.random as rd
import numpy.linalg as la
from numpy.linalg import multi_dot
from scipy.linalg import eigh
from quadprog import solve_qp

import pdb


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

    def __sub__(self, other):
        return Quadratic(self.matrix - other.matrix)

    def __matmul__(self, other):
        return self.matrix @ other.matrix

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
    return Quadratic(np.tile(block, (size_product, size_product)))


def random_psd(n):
    """generates a random  matrix
    """
    C = rd.rand(n, n)
    return Quadratic(C.T @ C)


def compute_minimal_elicitation(M_unsplit, M_split, P):
    P_perp = Quadratic(np.eye(P.dim)) - P

    alpha = M_unsplit.smallest_eigenvalue()

    tmp = multi_dot([P_perp.matrix, M_split.matrix, P_perp.matrix])
    symmetric_part = 0.5*(tmp + tmp.T)
    gamma, _ = eigh(symmetric_part, eigvals=(0, 0))

    tmp = multi_dot([P.matrix, M_split.matrix, P_perp.matrix])
    symmetric_part = 0.5*(tmp + tmp.T)
    beta, _ = eigh(symmetric_part, eigvals=(0, 0))
    beta = 0.5*beta

    trace_condition = -gamma - alpha
    det_condition = beta**2/alpha - gamma

    e_0 = max(trace_condition, det_condition)
    return e_0


def compute_minimal_elicitation_old(M_unsplit, M_split, P):
    P_perp = Quadratic(np.eye(P.dim)) - P

    alpha = M_unsplit.smallest_eigenvalue()
    gamma = la.norm(la.multi_dot([P_perp.matrix, M_split.matrix,
                    P_perp.matrix]), 2)
    beta = 0.5*la.norm(la.multi_dot([P.matrix, M_split.matrix+M_split.matrix.T,
                       P_perp.matrix]), 2)

    return beta**2/alpha - gamma
