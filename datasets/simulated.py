from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.sparse import rand as sprand
    from scipy.signal import fftconvolve
    from scipy.signal.windows import gaussian


def make_blur(size, std):
    gaussian_filter = np.outer(
        gaussian(size, std),
        gaussian(size, std))
    gaussian_filter /= gaussian_filter.sum()

    def blur_2d(u):
        return fftconvolve(u, gaussian_filter, mode="same")

    blur_2d.T = blur_2d
    blur_2d.norm = 1.

    return blur_2d


def identity(u):
    return u


identity.T = identity
identity.norm = 1.


class Dataset(BaseDataset):

    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # A * I + bruit ~ N(mu, sigma)
    parameters = {
        'std_noise': [0.3],
        'size_blur': [27],
        'std_blur': [2.],
        'subsampling': [4],
        'type_lin_op': ['deblurring', 'denoising'],
    }

    def __init__(self, std_noise=0.3,
                 size_blur=27, std_blur=8.,
                 subsampling=4,
                 random_state=27,
                 type_lin_op='denoising'):
        # Store the parameters of the dataset
        self.std_noise = std_noise
        self.size_blur = size_blur
        self.std_blur = std_blur
        self.subsampling = subsampling
        self.random_state = random_state
        self.type_lin_op = type_lin_op

    def set_lin_op(self):
        if self.type_lin_op == 'deblurring':
            blur_2d = make_blur(self.size_blur, self.std_blur)
            return blur_2d
        else:
            return identity

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        height, width = (64, 32)
        n_blocks = 10
        z = sprand(
            1, width, density=n_blocks/width,
            random_state=rng
        ).toarray()[0]
        img = (np.cumsum(rng.randn(height, width) * z, axis=1)
               [::self.subsampling, ::self.subsampling])
        lin_op = self.set_lin_op()
        y_degraded = (lin_op(img) +
                      rng.normal(0, self.std_noise, size=img.shape))
        data = dict(lin_op=lin_op, y=y_degraded)

        return data
