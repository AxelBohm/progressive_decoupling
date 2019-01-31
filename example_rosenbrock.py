import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from decoupling import decomposable_decoupling
from util import CallBack

import pdb


def main():

    n = 2
    x_opt = np.ones(n)

    # params
    a = 1
    b = 1
    niter = 30
    x0 = np.zeros(n)

    obj_fun = Rosenbrock(a, b)
    first_summand = RosenbrockFirstSummand(a, b)
    second_summand = RosenbrockSecondSummand(a, b)

    # hessian at the solution

    e = 0
    r = 0.01

    def dist_to_solution(x): return np.linalg.norm(x - x_opt)
    callback = CallBack(obj_fun, dist_to_solution)

    x_opt = decomposable_decoupling([first_summand, second_summand], niter,
                                    e, r, n, x0=x0, callback=callback)

    # x_opt = decomposable_decoupling([obj_fun], niter,
    #                                 e, r, n, x0=x0, callback=callback)

    pdb.set_trace()
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
        return (self.a - x[0])**2 + self.b*(x[1] - x[0]**2)**2

    def gradient(self, x):
        return np.array([2*(x[0] - self.a) + 4*self.b*(x[0]**2 - x[1])*x[0],
                         2*self.b*(x[1] - x[0]**2)])

    def prox(self, x, r):
        def prox_fun(w): return self.__call__(w) + r/2*np.linalg.norm(w-x)**2
        # res = minimize(prox_fun, x, method='nelder-mead')
        res = minimize(prox_fun, x, method='BFGS', jac=self.gradient)
        return res.x


class RosenbrockFirstSummand():

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, x):
        return (self.a - x[0])**2

    def prox(self, x, r):
        return np.array([(self.a+r*x[0])/(2+r), x[1]])


class RosenbrockSecondSummand():

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, x):
        return self.b*(x[1] - x[0]**2)**2

    def gradient(self, x):
        return np.array([4*self.b*(x[0]**2 - x[1])*x[0],
                        2*self.b*(x[1] - x[0]**2)])

    def prox(self, x, r):
        def prox_fun(w): return self.__call__(w) + r/2*np.linalg.norm(w-x)**2
        # res = minimize(prox_fun, x, method='nelder-mead')
        res = minimize(prox_fun, x, method='BFGS', jac=self.gradient)
        return res.x


if __name__ == "__main__":
    main()
