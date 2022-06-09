from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from PIL import Image, ImageOps
    import download
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

    def set_lin_op(self):
        if self.type_lin_op == 'deblurring':
            blur_2d = make_blur(self.size_blur, self.std_blur)
            return blur_2d
        else:
            return identity

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        img = download.download(
            "https://archive.org/download/SitaStills/01.RamShootsDemons.png",
            "./cartoon/01.RamShootsDemons", replace=False)
        img = Image.open(img)
        img = (np.array(ImageOps.grayscale(img))
               [::self.subsampling, ::self.subsampling]) / 255.0
        height, width = img.shape
        lin_op = self.set_lin_op()
        y_degraded = (lin_op(img) +
                      rng.normal(0, self.std_noise, size=(height, width)))
        data = dict(lin_op=lin_op, y=y_degraded)

        return data
