from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from PIL import Image, ImageOps
    import download
    from benchmark_utils.shared import make_blur


class Dataset(BaseDataset):

    name = "Inpainting"

    install_cmd = 'conda'
    requirements = ['pip:download']

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'std_noise': [0.02],
        'subsampling': [10],
        'type_A': ['inpainting'],
        'type_n': ['gaussian', 'laplace'],
        'random_state': [27]
    }

    def set_A(self, height):
        return make_blur(self.type_A, height)

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        img = download.download(
            "https://culturezvous.com/wp-content/uploads/2017/10/"
            "chateau-azay-le-rideau.jpg?download=true", replace=False)

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
