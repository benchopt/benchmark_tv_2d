from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from PIL import Image
    from scipy.signal import fftconvolve


class Dataset(BaseDataset):

    name = "Cartoon"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # A * I + bruit ~ N(mu, sigma)
    parameters = {
        'K': [50],
        'type_A': ['identity', 'diagonal', 'triangular', 'random']}

    def __init__(self, mu=0, sigma=0.3, K=10,
                 type_A='identity', random_state=27):
        # Store the parameters of the dataset
        self.mu = mu
        self.sigma = sigma
        self.K = K
        self.type_A = type_A
        self.random_state = random_state

    def set_A(self, rng):
        if self.type_A == 'diagonal':
            A = np.diag(rng.random(self.K))
        elif self.type_A == 'triangular':
            A = np.triu(rng.randn(self.K, self.K))
        elif self.type_A == 'random':
            A = rng.randn(self.K, self.K)
        else:
            A = np.eye(self.K, dtype=float)
        return A

    def get_data(self):
        height = 500
        width = 500
        rng = np.random.RandomState(47)
        y = np.array(
            Image.open('/mnt/share/\
            cartoon/01.RamShootsDemons_gray.png'))[:height, :width]
        mu = np.mean(y)
        sigma = np.std(y)
        A = self.set_A(rng)
        y_blurred = fftconvolve(y, A, mode="same") + \
            rng.normal(mu, sigma, size=(height, width))
        data = dict(A=A, y=y_blurred)

        return y_blurred.shape[0], data
