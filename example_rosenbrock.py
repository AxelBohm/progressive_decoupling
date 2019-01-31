import numpy as np
# import numpy.random as rd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from decoupling import decomposable_decoupling
from util import CallBack

# import pdb


def main():
    """TODO: solve min_x <x,Mx>
    by splitting M into A, C, and B
    M = (A   B)
        (B^t C)
    """
    n = 2

    # global minimizer
    x_opt = np.ones(n)

    # params
    e = 1
    r = 3
    niter = 10
    x0 = np.zeros(n)

    def dist_to_solution(x): return np.linalg.norm(x - x_opt)
    callback = CallBack(Rosenbrock, dist_to_solution)

    x_opt = decomposable_decoupling([RosenbrockFirstSummand,
                                    RosenbrockSecondSummand], niter, e, r, n,
                                    x0=x0, callback=callback)

    # make plots
    plt.plot(np.arange(niter), callback.stored_obj_fun_values)
    plt.xlabel('iterations')
    plt.ylabel('function value')
    plt.title(f'Rosenbrock Decoupling, r={r}, e={e}')
    plt.show()

    plt.plot(np.arange(niter), callback.stored_dist_to_solution)
    plt.xlabel('iterations')
    plt.ylabel('distance to solution')
    plt.title(f'Rosenbrock Decoupling, r={r}, e={e}')
    plt.show()


class Rosenbrock():

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, x):
        return (self.a - x)**2 + self.b*(x[1] - x[0]**2)**2

    def prox(self, x, r):
        pass


class RosenbrockFirstSummand():

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, x):
        return (self.a - x)**2

    def prox(self, x, r):
        return np.array([(self.a+r*x[0])/(2+r), x[1]])


class RosenbrockSecondSummand():

    def __inti__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, x):
        return self.b*(x[1] - x[0]**2)**2

    def prox(self, x, r):
        def prox_fun(w): return self.__call__(w) + r/2*np.linalg.norm(w-x)**2
        res = minimize(prox_fun, method='nelder-mead')
        return res.x


if __name__ == "__main__":
    main()
