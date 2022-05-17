from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.signal import fftconvolve
    import prox_tv as ptv


class Solver(BaseSolver):
    """Proximal gradient descent for analysis formulation."""
    name = 'PGD'

    install_cmd = 'conda'
    requirements = ['pip:prox-tv']

    stopping_strategy = "callback"

    # any parameter defined here is accessible as a class attribute

    def set_objective(self, A, reg, y):
        self.reg = reg
        self.A, self.y = A, y

    def run(self, callback):
        stepsize = 1 / (np.linalg.norm(self.A, ord=2)**2)  # 1/ rho
        # initialisation
        u = np.zeros((self.y.shape[0], self.y.shape[1]))
        while callback(u):
            u = ptv.tv1_2d(u - stepsize * fftconvolve(fftconvolve(u, self.A,
                           mode="same") - self.y, self.A[:, ::-1],
                           mode="same"),
                           self.reg * stepsize, method='condat')
        self.u = u

    def get_result(self):
        return self.u
