from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    """Chambolle-Pock (or PDHG) on higher dual (PD-Split)."""

    name = "Chambolle-Pock PD-split"

    stopping_criterion = SufficientProgressCriterion(
        patience=20,
        strategy="callback")

    # any parameter defined here is accessible as a class attribute
    parameters = {"ratio": [10.0],
                  "theta": [1.0]}

    def skip(self, lin_op, reg, y, isotropy):
        if isotropy not in ["anisotropic","isotropic"]:
            return True, "Only aniso and isoTV are implemented yet"
        return False, None

    def set_objective(self, lin_op, reg, y, isotropy):
        self.reg = reg
        self.isotropy = isotropy
        self.lin_op, self.y = lin_op, y

    def run(self, callback):
        # Block preconditioning (2x2)
        LD = np.sqrt(8.)  # Lipschitz constant associated to D
        LA = self.lin_op.norm
        tau = self.ratio / (LA + LD)
        sigma_v = 1.0 / (self.ratio * LD)
        sigma_w = 1.0 / (self.ratio * LA)
        # Init variables
        n, m = self.y.shape
        u = np.zeros((n, m))
        vh = np.zeros((n, m))  # we consider non-cyclic finite difference
        vv = np.zeros((n, m))
        w = np.zeros((n, m))
        u_bar = u
        proj = {
            'anisotropic': self._dual_prox_tv_aniso,
            'isotropic': self._dual_prox_tv_iso,
        }.get(self.isotropy, self._dual_prox_tv_aniso)

        while callback(u):
            u_old = u
            gh, gv = self._grad(u_bar)
            vh, vv = proj(vh + sigma_v * gh,
                          vv + sigma_v * gv)
            w_tmp = w + sigma_w * self.lin_op(u_bar)
            w = (w_tmp - sigma_w * self.y) / (1.0 + sigma_w)
            # grad.T = -div, hence + sign
            u = u + tau * self._div(vh, vv) - tau * self.lin_op.T(w)
            u_bar = u + self.theta * (u - u_old)
        self.u = u

    def get_result(self):
        return self.u

    def _div(self, vh, vv):
        dh = np.vstack((np.diff(vh, prepend=0, axis=0)[:-1,:], -vh[-1,:]))
        dv = np.column_stack((np.diff(vv, prepend=0, axis=1)[:,:-1], -vv[:,-1]))
        return dh + dv

    def _grad(self, u):
        # Neumann condition
        gh = np.pad(np.diff(u, axis=0), ((0, 1), (0, 0)), 'constant')
        gv = np.pad(np.diff(u, axis=1), ((0, 0), (0, 1)), 'constant')
        return gh, gv

    def _dual_prox_tv_aniso(self, vh, vv):
        return np.clip(vh, -self.reg, self.reg), \
            np.clip(vv, -self.reg, self.reg)

    def _dual_prox_tv_iso(self, vh, vv):
        norms = np.sqrt(vh ** 2 + vv ** 2)
        factors = 1. / np.maximum(1, norms)
        return vh * factors, vv * factors
