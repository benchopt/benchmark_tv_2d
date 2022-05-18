from benchopt import BaseObjective
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Objective(BaseObjective):
    name = "Total Variation 2D"

    parameters = {'reg': [0.5]}

    def __init__(self, reg=0.5):
        self.reg = reg  # 0<reg<1

    def set_data(self, lin_op, y):
        self.lin_op = lin_op
        self.y = y
        self.reg = self.reg

    def compute(self, u):
        # fftconvolve(u, self.A, mode="same")
        residual = self.y - self.lin_op(u)
        return .5 * np.linalg.norm(residual) ** 2 + \
            self.reg * self.isotropic_tv_value(u)

    def to_dict(self):
        return dict(lin_op=self.lin_op, reg=self.reg, y=self.y)

    def isotropic_tv_value(self, u):
        return np.sqrt(
            ((np.diff(u, axis=0))**2).sum() +
            ((np.diff(u, axis=1))**2).sum())

    def anisotropic_tv_value(self, u):
        return np.abs(np.diff(u, axis=0)).sum() +\
            np.abs(np.diff(u, axis=1)).sum()
