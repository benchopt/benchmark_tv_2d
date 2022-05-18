from benchopt import BaseObjective
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Objective(BaseObjective):
    name = "Total Variation 2D"

    parameters = {'reg': [0.5],
                  'isotropy': ["anisotropic", "isotropic", "split"]}

    def __init__(self, reg=0.5, isotropy="anisotropic"):
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
        else:                   # just aniso for the moment
            return lsq + \
                self.reg * self.anisotropic_tv_value(u)

    def to_dict(self):
        return dict(lin_op=self.lin_op,
                    reg=self.reg,
                    y=self.y,
                    isotropy=self.isotropy)

    def isotropic_tv_value(self, u):
        return np.sqrt(
            ((np.diff(u, axis=0))**2).sum() +
            ((np.diff(u, axis=1))**2).sum())

    def anisotropic_tv_value(self, u):
        return np.abs(np.diff(u, axis=0)).sum() +\
            np.abs(np.diff(u, axis=1)).sum()
