import numpy as np
import decoupling as dc
import example_matrix
import numpy.testing as npt
import pdb

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
    A = example_matrix.quadratic(np.array([[a11, 0.], [0., a22]]))
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
    A = example_matrix.quadratic(np.array([[a11, 0.], [0., a22]]))
    assert all(A.prox(np.zeros(2), 1.) == np.zeros(2))
    w = np.array([2, 3])
    r = 1.
    prox = A.prox(w, r)
    should_be = w.copy()
    should_be[0] = r/(a11+r)*should_be[0]
    should_be[1] = r/(a22+r)*should_be[1]
    npt.assert_almost_equal(prox, r/(a11+r)*w)


def test_example_matrix():
    example_matrix.main()
    assert True


def test_class():
    A = example_matrix.quadratic(np.array([[3, 2], [2, 1]]))
    x, fun_vals = dc.decomposable_decoupling([A], 10, 0, 1, 2, x0=np.ones(2))
