from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import prox_tv as ptv


class Solver(BaseSolver):
    """Proximal gradient descent for analysis formulation."""
    name = 'PGD'

    install_cmd = 'conda'
    requirements = ['pip:prox-tv']

    stopping_strategy = "callback"

    # any parameter defined here is accessible as a class attribute

    def set_objective(self, lin_op, reg, y):
        self.reg = reg
        self.lin_op, self.y = lin_op, y

    def run(self, callback):
        # initialisation
        n, m = self.y.shape[0], self.y.shape[1]
        u = np.zeros((n, m))
        stepsize = 1. / (self.lin_op.norm ** 2)
        while callback(u):
            u = ptv.tv1_2d(
                u - stepsize * self.lin_op.T(self.lin_op(u) - self.y),
                self.reg * stepsize, method='condat')
        self.u = u

    def get_result(self):
        return self.u
