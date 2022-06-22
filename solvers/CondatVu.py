from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientDescentCriterion
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    get_l2norm = import_ctx.import_from('shared', 'get_l2norm')


class Solver(BaseSolver):
    """Primal-Dual Splitting Method."""

    name = 'CondatVu'

    stopping_criterion = SufficientDescentCriterion(
        patience=3, strategy="callback"
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {'ratio': [1.],
                  'eta': [1.]}

    def set_objective(self, A, reg, delta, data_fit, y, isotropy):
        self.reg, self.delta = reg, delta
        self.isotropy = isotropy
        self.data_fit = data_fit
        self.A, self.y = A, y

    def run(self, callback):
        n, m = self.y.shape
        # Block preconditioning (2x2)
        LD = np.sqrt(8.)  # Lipschitz constant associated to D
        LA = get_l2norm(self.A)
        sigma_v = 1.0 / (self.ratio * LD)
        tau = 1 / (LA ** 2 / 2 + sigma_v * LD ** 2)
        eta = self.eta
        # initialisation
        u = np.zeros((n, m))
        vh = np.zeros((n, m))  # we consider non-cyclic finite difference
        vv = np.zeros((n, m))
        proj = {
            'anisotropic': self._dual_prox_tv_aniso,
            'isotropic': self._dual_prox_tv_iso,
        }.get(self.isotropy, self._dual_prox_tv_aniso)

        while callback(u):
            u_tmp = (u - tau * self.grad(u)
                     + tau * self._div(vh, vv))
            gh, gv = self._grad(2 * u_tmp - u)
            vh_tmp, vv_tmp = proj(vh + sigma_v * gh,
                                  vv + sigma_v * gv)
            u = eta * u_tmp + (1 - eta) * u
            vh = eta * vh_tmp + (1 - eta) * vh
            vv = eta * vv_tmp + (1 - eta) * vv
        self.u = u

    def get_result(self):
        return self.u

    def st(self, w, mu):
        w -= np.clip(w, -mu, mu)
        return w

    def grad(self, u):
        R = self.A @ u - self.y
        if self.data_fit == 'lsq':
            return self.A.T @ R
        else:
            return self.A.T @ self.grad_huber(R, self.delta)

    def grad_huber(self, R, delta):
        return np.where(np.abs(R) < delta, R, np.sign(R) * delta)

    def _div(self, vh, vv):
        dh = np.vstack(
            (np.diff(vh, prepend=0, axis=0)[:-1, :], -vh[-2, :])
        )
        dv = np.column_stack(
            (np.diff(vv, prepend=0, axis=1)[:, :-1], -vv[:, -2])
        )
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
        factors = 1. / np.maximum(1, 1./self.reg * norms)
        return vh * factors, vv * factors
