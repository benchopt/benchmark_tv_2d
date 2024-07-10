from benchopt import BaseObjective
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import torch
    import deepinv as dinv
    from benchmark_utils.shared import huber
    from benchmark_utils.matrix_op import grad


class Objective(BaseObjective):
    min_benchopt_version = "1.5"
    name = "TV2D"

    parameters = {'reg': [0.1, 0.2, 0.3, 0.4],
                  'delta': [0.9],
                  'isotropy': ["anisotropic", "isotropic"],
                  'data_fit': ["lsq", "huber"]}

    def linop(self, x2, size=False):
        x = torch.from_numpy(x2).unsqueeze(0)
        x = x.unsqueeze(0)
        if not size:
            size = x.shape
        if torch.cuda.is_available():
            device = dinv.utils.get_freer_gpu()
        else:
            device = 'cpu'
        operator = dinv.physics.Inpainting(
            tensor_size=size[1:],
            mask=0.5,
            device=device
        )
        out = operator(x).squeeze(0)
        out = out.squeeze(0)
        return out.numpy()

    def set_data(self, A, y):
        self.A = A
        self.y = y
        self.reg = self.reg

    def evaluate_result(self, u):
        if self.A != 0:
            R = self.y - self.A @ u  # residuals
        else:
            R = self.y - self.linop(u)

        if self.data_fit == "lsq":
            loss = .5 * np.linalg.norm(R) ** 2
        else:
            loss = huber(R, self.delta)

        if self.isotropy == "isotropic":
            penalty = self.isotropic_tv_value(u)
        else:
            penalty = self.anisotropic_tv_value(u)

        return loss + self.reg * penalty

    def get_one_result(self):
        return np.zeros(self.y.shape)

    def get_objective(self):
        return dict(A=self.A,
                    reg=self.reg,
                    delta=self.delta,
                    data_fit=self.data_fit,
                    y=self.y,
                    isotropy=self.isotropy)

    def isotropic_tv_value(self, u):
        gh, gv = grad(u)
        return (np.sqrt(gh ** 2 + gv ** 2)).sum()

    def anisotropic_tv_value(self, u):
        gh, gv = grad(u)
        return (np.abs(gh) + np.abs(gv)).sum()
