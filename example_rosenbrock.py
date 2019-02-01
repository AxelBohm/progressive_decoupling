import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import eigh

from decoupling import decomposable_decoupling
from util import CallBack

import pdb


def main():

    n = 2
    # param
    a = 1
    b = 100
    niter = 300
    x_opt = np.array([a, a**2])

    x0 = np.zeros(n)

    obj_fun = Rosenbrock(a, b)
    first_summand = RosenbrockFirstSummand(a, b)
    second_summand = RosenbrockSecondSummand(a, b)

    # hessian at the solution
    hessian_at_solution = obj_fun.hessian(x_opt)
    alpha, _ = eigh(hessian_at_solution, eigvals=(0, 0))
    P = np.ones((n, n))*1/n
    P_perp = np.eye(n) - P
    gamma = la.norm(la.multi_dot([P_perp, hessian_at_solution, P_perp]), 2)
    beta = 0.5*la.norm(la.multi_dot([P,
                       hessian_at_solution + hessian_at_solution.T,
                       P_perp]), 2)
    e_0 = beta**2/alpha + gamma

    e = e_0
    r = e+100

    def dist_to_solution(x): return np.linalg.norm(x - x_opt)
    callback = CallBack(obj_fun, dist_to_solution)

    x_opt = decomposable_decoupling([first_summand, second_summand], niter,
                                    e, r, n, x0=x0, callback=callback)

    pdb.set_trace()
    # x_opt = decomposable_decoupling([obj_fun], niter,
    #                                 e, r, n, x0=x0, callback=callback)

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

    def hessian(self, x):
        return np.array([[2 + 4*self.b*(x[0]**2 - x[1] + 2*x[0]),
                         -4*self.b*x[0]], [-4*self.b*x[0], 2*self.b]])

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
