from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientDescentCriterion
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.sparse.linalg import LinearOperator
    from scipy.optimize import minimize
    div = import_ctx.import_from('matrice_op', 'div')
    grad = import_ctx.import_from('matrice_op', 'grad')
    dual_prox_tv_aniso = import_ctx.import_from('matrice_op',
                                                'dual_prox_tv_aniso')
    dual_prox_tv_iso = import_ctx.import_from('matrice_op', 'dual_prox_tv_iso')


def loss(AtA, Aty, vh, vv, n, m, u):
    R = (AtA @ u).reshape((n, m)) - (Aty + div(vh, vv))
    return .5 * np.linalg.norm(R, ord=2) ** 2


def jac_loss(AtA, Aty, vh, vv, n, m, u):
    R = (AtA @ u).reshape((n, m)) - (Aty + div(vh, vv))
    return AtA @ (R.flatten())


class Solver(BaseSolver):
    """Dual Projected gradient descent for analysis formulation."""
    name = 'Dual PGD analysis'

    stopping_criterion = SufficientDescentCriterion(
        patience=3, strategy="callback"
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {'alpha': [1.],
                  'use_acceleration': [False, True]}

    def skip(self, A, reg, delta, data_fit, y, isotropy):
        if data_fit == 'huber':
            return True, "solver does not work with huber loss"
        elif max(y.shape) > 1e4:
            return True, "solver has to do a too large densification"
        elif isotropy not in ["anisotropic", "isotropic"]:
            return True, "Only aniso and isoTV are implemented yet"
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
        vh = np.zeros((n, m))  # we consider non-cyclic finite difference
        vv = np.zeros((n, m))
        LD = np.sqrt(8.)  # Lipschitz constant associated to D
        sigma_v = 1.0 / (10 * LD)
        tol_cg = 1e-12
        Aty = self.A.T @ self.y
        AtA = LinearOperator(shape=(n * m, n * m),
                             matvec=lambda x: self.A.T @ (
                                self.A @ x.reshape((n, m))))
        proj = {
            'anisotropic': dual_prox_tv_aniso,
            'isotropic': dual_prox_tv_iso,
        }.get(self.isotropy, dual_prox_tv_aniso)

        vh_old = vh.copy()
        vv_old = vv.copy()
        vh_acc = vh.copy()
        vv_acc = vv.copy()

        t_new = 1
        while callback(u):
            if self.use_acceleration:
                t_old = t_new
                t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
                vh_old[:] = vh
                vv_old[:] = vv
                vh[:] = vh_acc
                vv[:] = vv_acc

            gh, gv = grad(u)
            vh, vv = proj(vh + sigma_v * gh,
                          vv + sigma_v * gv,
                          self.reg)

            if self.use_acceleration:
                vh_acc[:] = vh + (t_old - 1.) / t_new * (vh - vh_old)
                vv_acc[:] = vv + (t_old - 1.) / t_new * (vv - vv_old)

            def func(u):
                return loss(AtA, Aty, vh, vv, n, m, u)

            def jac(u):
                return jac_loss(AtA, Aty, vh, vv, n, m, u)

            u = minimize(func, x0=u.flatten(), jac=jac,
                           method='BFGS', tol=tol_cg).x
            u = u.reshape((n, m))
        self.u = u

    def get_result(self):
        return self.u
