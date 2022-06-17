from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.sparse import rand as sprand
    from scipy.signal import fftconvolve
    from scipy.signal.windows import gaussian
    from scipy.sparse.linalg import LinearOperator


def make_blur(type_lin_op, size, std,
              height):
    if type_lin_op == 'denoising':
        lin_op = LinearOperator(
                dtype=np.float64,
                matvec=lambda x: x,
                matmat=lambda X: X,
                rmatvec=lambda x: x,
                rmatmat=lambda X: X,
                shape=(height, height),
            )
    elif type_lin_op == 'deblurring':
        filt = np.outer(
            gaussian(size, std),
            gaussian(size, std))
        filt /= filt.sum()
        lin_op = LinearOperator(
                dtype=np.float64,
                matvec=lambda x: fftconvolve(x, filt, mode='same'),
                matmat=lambda X: fftconvolve(X, filt, mode='same'),
                rmatvec=lambda x: fftconvolve(x, filt, mode='same'),
                rmatmat=lambda X: fftconvolve(X, filt, mode='same'),
                shape=(height, height),
            )
    return lin_op


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

    def set_lin_op(self, height):
        return make_blur(self.type_lin_op,
                         self.size_blur, self.std_blur,
                         height)

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
        lin_op = self.set_lin_op(img.shape[0])
        y_degraded = (lin_op @ img +
                      rng.normal(0, self.std_noise, size=img.shape))
        data = dict(lin_op=lin_op, y=y_degraded)

        return data
