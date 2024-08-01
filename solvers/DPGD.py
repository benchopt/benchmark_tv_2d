from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientDescentCriterion
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.sparse.linalg import LinearOperator
    from scipy.sparse.linalg import cg
    from benchmark_utils.matrix_op import div, grad
    from benchmark_utils.matrix_op import dual_prox_tv_iso
    from benchmark_utils.matrix_op import dual_prox_tv_aniso


class Solver(BaseSolver):
    """Dual Projected gradient descent for analysis formulation."""
    name = 'Dual PGD analysis'

    stopping_criterion = SufficientDescentCriterion(
        patience=3, strategy="callback"
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {'alpha': [1.],
                  'ratio': [10.],
                  'use_acceleration': [True]}

    def skip(self, A, reg, delta, data_fit, y, isotropy):
        if data_fit == 'huber':
            return True, "solver does not work with huber loss"
        elif max(y.shape) > 1e4:
            return True, "solver has to do a too large densification"
        elif isotropy not in ["anisotropic", "isotropic"]:
            return True, "Only aniso and isoTV are implemented yet"
        if (A @ y != y).all():
            return True, "solver only works for denoising"
        return False, None

    def set_objective(self, A, reg, delta, data_fit, y, isotropy):
        self.reg, self.delta = reg, delta
        self.isotropy = isotropy
        self.data_fit = data_fit
        self.A, self.y = A, y

    def run(self, callback):
        n, m = self.y.shape
        # initialisation
        u = np.zeros((n, m))
        self.u = u.copy()
        v = np.zeros((n, m))
        vh = np.zeros((n, m))  # we consider non-cyclic finite difference
        vv = np.zeros((n, m))
        LD = np.sqrt(8.)  # Lipschitz constant associated to D
        sigma_v = 1.0 / (self.ratio * LD)
        tol_cg = 1e-12
        Aty = self.A.T @ self.y
        AtA = LinearOperator(shape=(n * m, n * m),
                             matvec=lambda x: self.A.T @ (
            self.A @ x.reshape((n, m))))
        proj = {
            'anisotropic': dual_prox_tv_aniso,
            'isotropic': dual_prox_tv_iso,
        }.get(self.isotropy, dual_prox_tv_aniso)

        v_old = v.copy()
        vh_old = vh.copy()
        vv_old = vv.copy()
        v_acc = v.copy()
        vh_acc = vh.copy()
        vv_acc = vv.copy()

        t_new = 1
        while callback():
            if self.use_acceleration:
                t_old = t_new
                t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
                v_old[:] = v
                vh_old[:] = vh
                vv_old[:] = vv
                v[:] = v_acc
                vh[:] = vh_acc
                vv[:] = vv_acc

            v_tmp = (Aty + div(vh, vv)).flatten()
            v, _ = cg(AtA, v_tmp, x0=v.flatten(), tol=tol_cg)
            v = v.reshape((n, m))
            gh, gv = grad(v)
            vh, vv = proj(vh + sigma_v * gh,
                          vv + sigma_v * gv,
                          self.reg)

            if self.use_acceleration:
                v_acc[:] = v + (t_old - 1.) / t_new * (v - v_old)
                vh_acc[:] = vh + (t_old - 1.) / t_new * (vh - vh_old)
                vv_acc[:] = vv + (t_old - 1.) / t_new * (vv - vv_old)

            u_tmp = (Aty + div(vh, vv)).flatten()
            u, _ = cg(AtA, u_tmp, x0=u.flatten(), tol=tol_cg)
            u = u.reshape((n, m))
            self.u = u

    def get_result(self):
        return dict(u=self.u)
