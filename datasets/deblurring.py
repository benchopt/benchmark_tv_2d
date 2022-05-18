from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import misc
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


class Dataset(BaseDataset):

    name = "Deblurring"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # A * I + bruit ~ N(mu, sigma)
    parameters = {
        'std_noise': [0.3],
        'size_blur': [40],
        'std_blur': [8.],
    }

    def __init__(self, std_noise=0.3,
                 size_blur=40, std_blur=8.,
                 random_state=27):
        # Store the parameters of the dataset
        self.std_noise = std_noise
        self.std_blur = std_blur
        self.random_state = random_state

    def set_lin_op(self):
        blur_2d = make_blur(self.size_blur, self.std_blur)
        return blur_2d

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        img = misc.face(gray=True)[::4, ::4]
        height, width = img.shape
        lin_op = self.set_lin_op()
        y_degraded = lin_op(img) + \
            rng.normal(0, self.std_noise, size=(height, width))
        data = dict(lin_op=lin_op, y=y_degraded)

        return y_degraded.shape[0], data
