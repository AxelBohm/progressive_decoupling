import numpy as np
import numpy.random as rd
import decoupling as dc
import example_matrix_indefinite as emi
import example_matrix_convex as emc
import example_rosenbrock
import util
import numpy.testing as npt
import pdb


def test_projection():
    P = util.projection_onto_linkage_space(2, 3)
    P_correct = 1/3*np.array([[1, 0, 1, 0, 1, 0],
                              [0, 1, 0, 1, 0, 1],
                              [1, 0, 1, 0, 1, 0],
                              [0, 1, 0, 1, 0, 1],
                              [1, 0, 1, 0, 1, 0],
                              [0, 1, 0, 1, 0, 1],
                              [1, 0, 1, 0, 1, 0]])

    npt.assert_almost_equal(P.matrix, P_correct)


def test_decomposable_decoupling():
    """
    The splitting case. This means that the linkage is given by the subspace
    S = span (1,1,1,...).
    """
    n = 100
    x0 = np.zeros(n)
    y0 = np.zeros(n)


def test_prox():
    a11 = 1.
    a22 = 1.
    A = util.Quadratic(np.array([[a11, 0.], [0., a22]]))
    assert all(A.prox(np.zeros(2), 1.) == np.zeros(2))
    w = np.array([2., 3.])
    rr = 1.
    prox = A.prox(w, rr)
    should_be = w.copy()
    should_be[0] = rr/(a11+rr)*w[0]
    should_be[1] = rr/(a22+rr)*w[1]
    npt.assert_almost_equal(prox, should_be)


def test_prox2():
    a11 = 2.
    a22 = 3.
    A = util.Quadratic(np.array([[a11, 0.], [0., a22]]))
    assert all(A.prox(np.zeros(2), 1.) == np.zeros(2))
    w = np.array([2., 3.])
    r = 1.
    prox = A.prox(w, r)
    should_be = w.copy()
    should_be[0] = r/(a11+r)*should_be[0]
    should_be[1] = r/(a22+r)*should_be[1]
    npt.assert_almost_equal(prox, should_be)


def test_matrix_split():
    n = 100
    nA = 4
    # nC = 60
    M = rd.rand(n, n)
    M = np.dot(M, M.transpose())
    M_obj = util.Quadratic(M)
    A = M.copy()
    A[nA:n, :] = 0
    A[:, nA:n] = 0
    A = util.Quadratic(A)
    C = M.copy()
    C[0:nA, :] = 0
    C[:, 0:nA] = 0
    C = util.Quadratic(C)
    B = M.copy()
    B[0:nA, 0:nA] = 0
    B[nA:n, nA:n] = 0
    B = util.Quadratic(M)

    # global minimizer
    x_opt = np.zeros(n)

    # params
    e = 1
    r = 2
    niter = 10
    x0 = np.ones(n)

    x_opt = dc.decomposable_decoupling([A, C, B], niter, e, r, n, x0=x0)

    # should actually be zero
    assert np.max(x_opt) <= 1
    # should be zero
    assert M_obj(x_opt) <= 1


def test_example_matrix_indefinite():
    emi.main()


def test_example_matrix_convex():
    emc.main()


def test_class_matrix():
    A = util.Quadratic(np.array([[3, 2], [2, 1]]))
    x, fun_vals = dc.decomposable_decoupling([A], 10, 0, 1, 2, x0=np.ones(2))


def test_rosenbrock():
    example_rosenbrock.main()


def test_class_rosenbrock():
    R = example_rosenbrock.Rosenbrock(1, 100)
    npt.assert_almost_equal(R(np.ones(2)), 0)

    R = example_rosenbrock.Rosenbrock(2, 10)
    npt.assert_almost_equal(R(np.array([2, 4])), 0)

    R1 = example_rosenbrock.RosenbrockFirstSummand(1, 100)
    npt.assert_almost_equal(R1(np.ones(2)), 0)
