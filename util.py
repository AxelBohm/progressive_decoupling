
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
