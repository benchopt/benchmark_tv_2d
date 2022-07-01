from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientDescentCriterion
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.sparse.linalg import LinearOperator
    from scipy.sparse.linalg import cg
    div = import_ctx.import_from('matrice_op', 'div')
    grad = import_ctx.import_from('matrice_op', 'grad')
    dual_prox_tv_aniso = import_ctx.import_from('matrice_op',
                                                'dual_prox_tv_aniso')
    dual_prox_tv_iso = import_ctx.import_from('matrice_op', 'dual_prox_tv_iso')


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
        v_tmp = np.zeros((n, m))
        # alpha / rho
        DA_inv = grad(np.linalg.pinv(self.A @ np.identity(n)))[0]
        stepsize = self.alpha / (np.linalg.norm(DA_inv, ord=2)**2)
        tol_cg = 1e-12
        Aty = self.A.T @ self.y
        AtA = LinearOperator(shape=(n * m, n * m),
                             matvec=lambda x: self.A.T @ (
                                self.A @ (x.reshape((n, m)))))
        proj = {
            'anisotropic': dual_prox_tv_aniso,
            'isotropic': dual_prox_tv_iso,
        }.get(self.isotropy, dual_prox_tv_aniso)

        while callback(u):
            v_tmp, _ = cg(AtA, (Aty + div(vh, vv)).flatten(),
                          x0=v_tmp.flatten(), tol=tol_cg)
            v_tmp = v_tmp.reshape((n, m))
            vh, vv = proj(vh + stepsize * grad(v_tmp)[0],
                          vv + stepsize * grad(v_tmp)[1],
                          self.reg)
            u, _ = cg(AtA, (Aty + div(vh, vv)).flatten(),
                      x0=u.flatten(), tol=tol_cg)
            u = u.reshape((n, m))
        self.u = u

    def get_result(self):
        return self.u
