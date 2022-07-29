from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientDescentCriterion
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    get_l2norm = import_ctx.import_from('shared', 'get_l2norm')
    grad_F = import_ctx.import_from('shared', 'grad_F')
    div = import_ctx.import_from('matrix_op', 'div')
    grad = import_ctx.import_from('matrix_op', 'grad')
    dual_prox_tv_aniso = import_ctx.import_from('matrix_op',
                                                'dual_prox_tv_aniso')
    dual_prox_tv_iso = import_ctx.import_from('matrix_op', 'dual_prox_tv_iso')


class Solver(BaseSolver):
    """Primal-Dual Splitting Method."""

    name = 'CondatVu'

    stopping_criterion = SufficientDescentCriterion(
        patience=3, strategy="callback"
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {'ratio': [1.],
                  'eta': [1.]}

    def skip(self, A, reg, delta, data_fit, y, isotropy):
        if isotropy not in ["anisotropic", "isotropic"]:
            return True, "Only aniso and isoTV are implemented yet"
        return False, None

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
            'anisotropic': dual_prox_tv_aniso,
            'isotropic': dual_prox_tv_iso,
        }.get(self.isotropy, dual_prox_tv_aniso)

        while callback(u):
            u_tmp = (u - tau * grad_F(self.y, self.A, u,
                                      self.data_fit, self.delta)
                     + tau * div(vh, vv))
            gh, gv = grad(2 * u_tmp - u)
            vh_tmp, vv_tmp = proj(vh + sigma_v * gh,
                                  vv + sigma_v * gv,
                                  self.reg)
            u = eta * u_tmp + (1 - eta) * u
            vh = eta * vh_tmp + (1 - eta) * vh
            vv = eta * vv_tmp + (1 - eta) * vv
        self.u = u

    def get_result(self):
        return self.u
