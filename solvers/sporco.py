from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from sporco.admm import tvl2
    get_l2norm = import_ctx.import_from('shared', 'get_l2norm')
    grad_F = import_ctx.import_from('shared', 'grad_F')


class Solver(BaseSolver):
    """SParse Optimization Research COde (ADMM)"""
    name = 'sporco'

    install_cmd = 'conda'
    # We need blas devel to get the include file for BLAS/LAPACK operations
    requirements = ["blas-devel", 'pip:sporco']

    stopping_strategy = 'callback'

    parameters = {'use_acceleration': [False, True]}

    def skip(self, A, reg, delta, data_fit, y, isotropy):
        if isotropy != "isotropic":
            return True, "prox-tv supports only isoTV"
        return False, None

    def set_objective(self, A, reg, delta, data_fit, y, isotropy):
        self.reg, self.delta = reg, delta
        self.isotropy = isotropy
        self.data_fit = data_fit
        self.A, self.y = A, y

    def run(self, callback):
        n, m = self.y.shape
        # initialisation
        stepsize = 1. / (get_l2norm(self.A) ** 2)
        u = np.zeros((n, m))
        u_acc = u.copy()
        u_old = u.copy()

        t_new = 1
        opt = tvl2.TVL2Denoise.Options({'FastSolve': True,
                                        'Verbose': False,
                                        'MaxMainIter': 100,
                                        'GSTol': 1e-12,
                                        'AbsStopTol': 1e-12,
                                        'gEvalY': False,
                                        'AutoRho': {'AutoScaling': False,
                                                    'Enabled': False}})

        while callback(u):
            if self.use_acceleration:
                t_old = t_new
                t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
                u_old[:] = u
                u[:] = u_acc
            u = tvl2.TVL2Denoise(
                S=u - stepsize * grad_F(self.y, self.A, u,
                                        self.data_fit, self.delta),
                lmbda=self.reg * stepsize, opt=opt,
                axes=(0, 1), caxis=None).solve()
            if self.use_acceleration:
                u_acc[:] = u + (t_old - 1.) / t_new * (u - u_old)
        self.u = u

    def get_result(self):
        return self.u
