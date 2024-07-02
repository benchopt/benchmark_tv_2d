from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import deepinv as dinv
    import torch
    from deepinv.optim.data_fidelity import L2


class Solver(BaseSolver):
    name = 'deepinv'

    parameters = {
        'recon': ['TV', 'DRUNet']
    }

    def skip(self, A, reg, delta, data_fit, y, isotropy):
        if data_fit == 'huber':
            return True, f"solver does not work with {data_fit} loss"
        elif isotropy == 'anisotropic':
            return True, f"solver does not work with {isotropy} regularization"
        return False, None

    def set_objective(self, A, reg, delta, data_fit, y, isotropy):
        self.A, self.reg, self.y = A, reg, torch.from_numpy(y)
        self.delta, self.data_fit = delta, data_fit
        self.isotropy = isotropy

    def run(self, n_iter):
        if torch.cuda.is_available():
            device = dinv.utils.get_freer_gpu()
        else:
            device = 'cpu'

        y = self.y
        reg = self.reg
        x = y.clone().to(device)
        x = x.unsqueeze(0)
        x = x.unsqueeze(0)
        data_fidelity = L2()
        if self.recon == 'TV':
            prior = dinv.optim.TVPrior()
        elif self.recon == 'DRUNet':
            denoiser = dinv.models.DRUNet(
                in_channels=1, out_channels=1, pretrained='download',
                device=device)
            prior = dinv.optim.PnP(denoiser=denoiser)

        physics = dinv.physics.Inpainting(
            tensor_size=x.shape[1:],
            mask=0.5,
            device=device
        )
        physics.noise_model = dinv.physics.GaussianNoise(sigma=0.2)
        for _ in range(n_iter):
            x = x - data_fidelity.grad(x, y, physics)
            if self.recon == 'TV':
                x = prior.prox(x,  gamma=reg)
            elif self.recon == 'DRUNet':
                x = denoiser(x, reg)
        self.out = x.clone()
        self.out = self.out.squeeze(0)
        self.out = self.out.squeeze(0)

    def get_result(self):
        return dict(u=self.out.numpy())
