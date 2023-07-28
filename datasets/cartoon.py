from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from PIL import Image, ImageOps
    import download
    from benchmark_utils.shared import make_blur


class Dataset(BaseDataset):

    name = "Cartoon"

    install_cmd = 'conda'
    requirements = ['pip:download']

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'std_noise': [0.02],
        'size_blur': [27],
        'std_blur': [2.],
        'subsampling': [10],
        'type_A': ['deblurring', 'denoising'],
        'type_n': ['gaussian', 'laplace']
    }

    def __init__(self, std_noise=0.02,
                 size_blur=27, std_blur=8.,
                 subsampling=4,
                 random_state=27,
                 type_A='denoising',
                 type_n='gaussian'):
        self.std_noise = std_noise
        self.size_blur = size_blur
        self.std_blur = std_blur
        self.subsampling = subsampling
        self.random_state = random_state
        self.type_A, self.type_n = type_A, type_n

    def set_A(self, height):
        return make_blur(self.type_A, height,
                         self.size_blur, self.std_blur)

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        img = download.download(
            "https://archive.org/download/SitaStills/01.RamShootsDemons.png",
            "./cartoon/01.RamShootsDemons", replace=False)
        img = Image.open(img)
        img = (np.array(ImageOps.grayscale(img))
               [::self.subsampling, ::self.subsampling]) / 255.0
        height, width = img.shape
        if self.type_n == 'gaussian':
            n = rng.normal(0, self.std_noise, size=(height, width))
        elif self.type_n == 'laplace':
            n = rng.laplace(0, self.std_noise, size=(height, width))
        A = self.set_A(height)
        y_degraded = A @ img + n

        return dict(A=A, y=y_degraded)
