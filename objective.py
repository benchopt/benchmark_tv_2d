from benchopt import BaseObjective
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.signal import fftconvolve


class Objective(BaseObjective):
    name = "Ordinary Least Squares"

    parameters = {'reg': [0.5]}

    def __init__(self, reg=0.5):
        self.reg = reg  # 0<reg<1

    def set_data(self, A, y):
        self.A = A
        self.y = y
        # height = self.y.shape[0]
        # width = self.y.shape[1]
        # L = np.tri(height)
        # AL = self.A @ L
        # S = self.A.T @ np.ones((height, width))
        # c = np.linalg.norm((S.T @ self.y)/(S.T @ S), ord=2)
        # self.c = c
        # z = np.zeros((height, width))
        # z[0][0] = c
        # reg_max = np.max(abs(- AL.T @ (self.y - AL @ z)))
        reg_max = 100
        self.reg = self.reg*reg_max

    def compute(self, u):
        R = self.y - fftconvolve(u, self.A, mode="same")
        return .5 * np.linalg.norm(R) + \
            self.reg*np.sqrt(((np.diff(u, axis=0))**2).sum() +
                             ((np.diff(u, axis=1))**2).sum())

    def to_dict(self):
        return dict(A=self.A, reg=self.reg, y=self.y)
