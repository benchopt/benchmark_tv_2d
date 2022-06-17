from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import misc
    from scipy.signal import fftconvolve
    from scipy.signal.windows import gaussian
    from scipy.sparse.linalg import LinearOperator


def make_blur(size, std, height):
    gaussian_filter = np.outer(
        gaussian(size, std),
        gaussian(size, std))
    gaussian_filter /= gaussian_filter.sum()
    lin_op = LinearOperator(
        dtype=np.float64,
        matvec=lambda x: fftconvolve(x, gaussian_filter, mode='same'),
        matmat=lambda X: fftconvolve(X, gaussian_filter, mode='same'),
        rmatvec=lambda x: fftconvolve(x, gaussian_filter, mode='same'),
        rmatmat=lambda X: fftconvolve(X, gaussian_filter, mode='same'),
        shape=(height, height),
    )
    return lin_op


class Dataset(BaseDataset):

    name = "Deblurring"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # A * I + bruit ~ N(mu, sigma)
    parameters = {
        'std_noise': [0.02],
        'size_blur': [27],
        'std_blur': [2.],
        'subsampling': [4],
    }

    def __init__(self, std_noise=0.3,
                 size_blur=27, std_blur=8.,
                 subsampling=4,
                 random_state=27):
        # Store the parameters of the dataset
        self.std_noise = std_noise
        self.size_blur = size_blur
        self.std_blur = std_blur
        self.subsampling = subsampling
        self.random_state = random_state

    def set_lin_op(self, height):
        blur_2d = make_blur(self.size_blur, self.std_blur, height)
        return blur_2d

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        img = misc.face(gray=True)[::self.subsampling, ::self.subsampling]
        img = img / 255.0
        height, width = img.shape
        lin_op = self.set_lin_op(height)
        y_degraded = (lin_op @ img +
                      rng.normal(0, self.std_noise, size=(height, width)))
        data = dict(lin_op=lin_op, y=y_degraded)

        return data
