from benchopt import BaseObjective
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Objective(BaseObjective):
    name = "TV2D"

    parameters = {'reg': [0.02],
                  'isotropy': ["anisotropic", "isotropic", "split"]}

    def __init__(self, reg=0.02, isotropy="anisotropic"):
        self.reg = reg
        self.isotropy = isotropy

    def set_data(self, lin_op, y):
        self.lin_op = lin_op
        self.y = y
        self.reg = self.reg

    def compute(self, u):
        residual = self.y - self.lin_op(u)
        lsq = .5 * np.linalg.norm(residual) ** 2
        if self.isotropy == "isotropic":
            return lsq + \
                self.reg * self.isotropic_tv_value(u)
        else:
            return lsq + \
                self.reg * self.anisotropic_tv_value(u)

    def get_one_solution(self):
        return np.zeros(self.y.shape)

    def to_dict(self):
        return dict(lin_op=self.lin_op,
                    reg=self.reg,
                    y=self.y,
                    isotropy=self.isotropy)

    def isotropic_tv_value(self, u):
        gh, gv = self.grad(u)
        return (np.sqrt(gh ** 2 + gv ** 2)).sum()

    def anisotropic_tv_value(self, u):
        gh, gv = self.grad(u)
        return (np.abs(gh) + np.abs(gv)).sum()

    def grad(self, u):
        # Neumann condition
        gh = np.pad(np.diff(u, axis=0), ((0, 1), (0, 0)), 'constant')
        gv = np.pad(np.diff(u, axis=1), ((0, 0), (0, 1)), 'constant')
        return gh, gv
