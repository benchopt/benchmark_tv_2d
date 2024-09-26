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
    requirements = (
        ["pip:git+https://github.com/bwohlberg/sporco.git"]
    )

    stopping_strategy = 'callback'

    def skip(self, A, reg, delta, data_fit, y, isotropy):
        if isotropy != "isotropic":
            return True, "sporco supports only isoTV"
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
        opt = tvl2.TVL2Denoise.Options({'FastSolve': True,
                                        'Verbose': False,
                                        'MaxMainIter': 100,
                                        'AbsStopTol': 0.,
                                        'RelStopTol': 0.,
                                        'RelaxParam': 0.5,
                                        'AutoRho': {'AutoScaling': False,
                                                    'Enabled': False},
                                        'gEvalY': False,
                                        'MaxGSIter': 100,
                                        'GSTol': 0.})

        while callback(u):
            u = tvl2.TVL2Denoise(
                S=u - stepsize * grad_F(self.y, self.A, u,
                                        self.data_fit, self.delta),
                lmbda=self.reg * stepsize, opt=opt,
                axes=(0, 1), caxis=None).solve()
        self.u = u

    def get_result(self):
        return self.u
