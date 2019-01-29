import numpy as np
import decoupling as dc
import example_matrix


def test_decomposable_decoupling():
    """
    The splitting case. This means that the linkage is given by the subspace
    S = span (1,1,1,...).
    """
    n = 100
    x0 = np.zeros(n)
    y0 = np.zeros(n)


def test_example_matrix():
    example_matrix.main()
    assert True
