from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import prox_tv as ptv
    get_l2norm = import_ctx.import_from('shared', 'get_l2norm')


class Solver(BaseSolver):
    """Primal forward-backward for anisoTV using prox-tv."""
    name = 'PGD'

    install_cmd = 'conda'
    # We need blas devel to get the include file for BLAS/LAPACK operations
    requirements = ["blas-devel", 'pip:prox-tv']

    stopping_strategy = "callback"

    parameters = {'prox_tv_method': [
        "dr",
        "pd",
        "yang",
        "condat",
        "chambolle-pock",
        "kolmogorov"
    ]}

    def skip(self, A, reg, delta, data_fit, y, isotropy):
        if isotropy != "anisotropic":
            return True, "prox-tv supports only anisoTV"
        return False, None

    def set_objective(self, A, reg, delta, data_fit, y, isotropy):
        self.reg, self.delta = reg, delta
        self.isotropy = isotropy
        self.data_fit = data_fit
        self.A, self.y = A, y

    def run(self, callback):
        n, m = self.y.shape[0], self.y.shape[1]
        u = np.zeros((n, m))
        stepsize = 1. / (get_l2norm(self.A) ** 2)
        while callback(u):
            u = ptv.tv1_2d(
                u - stepsize * self.grad(u),
                self.reg * stepsize, method=self.prox_tv_method)
        self.u = u

    def get_result(self):
        return self.u

    def grad(self, u):
        R = self.A(u) - self.y
        if self.data_fit == 'lsq':
            return self.A.T @ R
        else:
            return self.A.T @ self.grad_huber(R, self.delta)

    def grad_huber(self, R, delta):
        return np.where(np.abs(R) < delta, R, np.sign(R) * delta)
