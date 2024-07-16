from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from benchmark_utils.shared import get_l2norm
    from benchmark_utils.matrix_op import div, grad
    from benchmark_utils.matrix_op import prox_huber
    from benchmark_utils.matrix_op import dual_prox_tv_iso
    from benchmark_utils.matrix_op import dual_prox_tv_aniso


class Solver(BaseSolver):
    """Chambolle-Pock (or PDHG) on higher dual (PD-Split)."""

    name = "Chambolle-Pock PD-split"

    stopping_criterion = SufficientProgressCriterion(
        patience=3,
        strategy="callback")

    # any parameter defined here is accessible as a class attribute
    parameters = {"ratio": [10.0],
                  "eta": [1.0]}

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
        # Block preconditioning (2x2)
        LD = np.sqrt(8.)  # Lipschitz constant associated to D
        LA = get_l2norm(self.A)
        tau = self.ratio / (LA + LD)
        sigma_v = 1.0 / (self.ratio * LD)
        sigma_w = 1.0 / (self.ratio * LA)
        # Init variables
        n, m = self.y.shape
        self.u = u = np.zeros((n, m))
        vh = np.zeros((n, m))  # we consider non-cyclic finite difference
        vv = np.zeros((n, m))
        w = np.zeros((n, m))
        u_bar = u
        proj = {
            'anisotropic': dual_prox_tv_aniso,
            'isotropic': dual_prox_tv_iso,
        }.get(self.isotropy, dual_prox_tv_aniso)

        while callback():
            u_old = u
            gh, gv = grad(u_bar)
            vh, vv = proj(vh + sigma_v * gh,
                          vv + sigma_v * gv,
                          self.reg)
            w_tmp = w + sigma_w * self.A @ u_bar
            if self.data_fit == "huber":
                # Use Moreau identity + translation rule
                prox_out = prox_huber(
                    w_tmp / sigma_w - self.y, 1.0 / sigma_w,
                    self.delta
                )
                w = w_tmp - sigma_w * (prox_out + self.y)
            else:
                w = (w_tmp - sigma_w * self.y) / (1.0 + sigma_w)
            # grad.T = -div, hence + sign
            u = u + tau * div(vh, vv) - tau * self.A.T @ w
            u_bar = u + self.eta * (u - u_old)
            self.u = u

    def get_result(self):
        return dict(u=self.u)
