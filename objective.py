from benchopt import BaseObjective
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Objective(BaseObjective):
    name = "TV2D"

    parameters = {'reg': [0.02],
                  'delta': [0.9],
                  'isotropy': ["anisotropic", "isotropic", "split"],
                  'data_fit': ["lsq", "huber"]}

    def __init__(self, reg=0.02, delta=0.1,
                 isotropy="anisotropic", data_fit="lsq"):
        self.reg = reg
        self.delta = delta
        self.isotropy = isotropy
        self.data_fit = data_fit

    def set_data(self, lin_op, y):
        self.lin_op = lin_op
        self.y = y
        self.reg = self.reg

    def compute(self, u):
        residual = self.y - self.lin_op(u)
        if self.data_fit == "lsq":
            loss = .5 * np.linalg.norm(residual) ** 2
        else:
            loss = self.huber(residual, self.delta)
        if self.isotropy == "isotropic":
            penality = self.isotropic_tv_value(u)
        else:
            penality = self.anisotropic_tv_value(u)
        return loss + self.reg * penality

    def get_one_solution(self):
        return np.zeros(self.y.shape)

    def to_dict(self):
        return dict(lin_op=self.lin_op,
                    reg=self.reg,
                    delta=self.delta,
                    data_fit=self.data_fit,
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

    def huber(self, R, delta):
        norm_1 = np.abs(R)
        loss = np.where(norm_1 < delta,
                        0.5 * norm_1**2,
                        delta * norm_1 - 0.5 * delta**2)
        return np.sum(loss)
