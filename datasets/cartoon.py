from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from PIL import Image, ImageOps
    import download
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

    name = "Cartoon"

    install_cmd = 'conda'
    requirements = ['pip:download']

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # A * I + bruit ~ N(mu, sigma)
    parameters = {
        'std_noise': [0.02],
        'size_blur': [27],
        'std_blur': [2.],
        'subsampling': [10],
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
        img = download.download(
            "https://archive.org/download/SitaStills/01.RamShootsDemons.png",
            "./cartoon/01.RamShootsDemons", replace=False)
        img = Image.open(img)
        img = (np.array(ImageOps.grayscale(img))
               [::self.subsampling, ::self.subsampling]) / 255.0
        height, width = img.shape
        lin_op = self.set_lin_op(height)
        y_degraded = (lin_op @ img +
                      rng.normal(0, self.std_noise, size=(height, width)))
        data = dict(lin_op=lin_op, y=y_degraded)

        return data
