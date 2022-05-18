from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import prox_tv as ptv


class Solver(BaseSolver):
    """Primal forward-backward for anisoTV using prox-tv."""
    name = 'PGD'

    install_cmd = 'conda'
    requirements = ['pip:prox-tv']

    stopping_strategy = "callback"

    parameters = {'prox_tv_method': [
        "dr",
        "pd",
        "yang",
        "condat",
        "chambolle-pock",
        "kolmogorov"
    ]}

    def set_objective(self, lin_op, reg, y, isotropy):
        self.reg = reg
        self.isotropy = isotropy
        self.lin_op, self.y = lin_op, y

    def skip(self, lin_op, reg, y, isotropy):
        if isotropy != "anisotropic":
            return True, "prox-tv supports only anisoTV"
        return False, None

    def run(self, callback):
        n, m = self.y.shape[0], self.y.shape[1]
        u = np.zeros((n, m))
        stepsize = 1. / (self.lin_op.norm ** 2)
        while callback(u):
            u = ptv.tv1_2d(
                u - stepsize * self.lin_op.T(self.lin_op(u) - self.y),
                self.reg * stepsize, method=self.prox_tv_method)
        self.u = u

    def get_result(self):
        return self.u
