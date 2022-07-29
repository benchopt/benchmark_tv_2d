from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientDescentCriterion
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.sparse.linalg import LinearOperator
    from scipy.sparse.linalg import cg
    huber = import_ctx.import_from('shared', 'huber')
    grad_huber = import_ctx.import_from('shared', 'grad_huber')
    div = import_ctx.import_from('matrix_op', 'div')
    grad = import_ctx.import_from('matrix_op', 'grad')
    dual_prox_tv_aniso = import_ctx.import_from('matrix_op',
                                                'dual_prox_tv_aniso')
    dual_prox_tv_iso = import_ctx.import_from('matrix_op', 'dual_prox_tv_iso')


class Solver(BaseSolver):
    """Alternating direction method."""
    name = 'ADMM'

    stopping_criterion = SufficientDescentCriterion(
        patience=3, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {'gamma': [0.1]}

    def skip(self, A, reg, delta, data_fit, y, isotropy):
        if data_fit == 'huber':
            return True, "solver does not work with huber loss"
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
        zh = np.zeros((n, m))  # we consider non-cyclic finite difference
        zv = np.zeros((n, m))
        muh = np.zeros((n, m))  # we consider non-cyclic finite difference
        muv = np.zeros((n, m))
        # prox of sigma * G*, where G* is conjugate of G
        # G is reg * l1-norm
        proj = {
            'anisotropic': dual_prox_tv_aniso,
            'isotropic': dual_prox_tv_iso,
        }.get(self.isotropy, dual_prox_tv_aniso)

        gamma = self.gamma
        tol_cg = 1e-12
        Aty = self.A.T @ self.y
        # D @ x = grad(x)
        # D.T @ x = - div(xh, xv)
        AtA_gDtD = LinearOperator(shape=(n*m, n*m),
                                  matvec=lambda x: self.A.T @ (
                                      self.A @ x.reshape((n, m)))
                                  - gamma * div(grad(x.reshape((n, m)))[0],
                                                grad(x.reshape((n, m)))[1]))
        while callback(u):
            u_tmp = (Aty + div(muh, muv) - gamma * div(zh, zv)).flatten()
            u, _ = cg(AtA_gDtD, u_tmp, x0=u.flatten(), tol=tol_cg)
            u = u.reshape((n, m))
            gh, gv = grad(u)
            zh, zv = proj(gh * gamma + muh,
                          gv * gamma + muv,
                          self.reg)
            zh = (gh * gamma + muh - zh) / gamma
            zv = (gv * gamma + muv - zv) / gamma
            muh += gamma * (gh - zh)
            muv += gamma * (gv - zv)
        self.u = u

    def get_result(self):
        return self.u
