from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from PIL import Image
    from scipy.signal import fftconvolve
    from scipy.signal.windows import gaussian


GAUSSIAN_FILTER = np.outer(
    gaussian(64, 8),
    gaussian(64, 8))
GAUSSIAN_FILTER /= np.linalg.norm(GAUSSIAN_FILTER, ord=2)


def blur_2d(u):
    return fftconvolve(u, GAUSSIAN_FILTER, mode="same")


blur_2d.T = blur_2d
blur_2d.norm = 1.


def identity(u):
    return u


identity.T = identity
identity.norm = 1.


class Dataset(BaseDataset):

    name = "Cartoon"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # A * I + bruit ~ N(mu, sigma)
    parameters = {
        'scenario': ['denoising', 'deblurring']}

    def __init__(self, std_noise=0.3,
                 scenario='denoising', random_state=27):
        # Store the parameters of the dataset
        self.std_noise = std_noise
        self.scenario = scenario
        self.random_state = random_state

    def set_lin_op(self):
        if self.scenario == 'deblurring':
            lin_op = blur_2d
        else:
            lin_op = identity
        return lin_op

    def get_data(self):
        rng = np.random.RandomState(47)
        img = np.array(
            Image.open('img.png')).mean(axis=2)
        height, width = img.shape
        lin_op = self.set_lin_op()
        y_degraded = lin_op(img) + \
            rng.normal(0, self.std_noise, size=(height, width))
        data = dict(lin_op=lin_op, y=y_degraded)

        return y_degraded.shape[0], data
