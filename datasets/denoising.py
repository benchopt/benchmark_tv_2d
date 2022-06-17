from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import misc
    from scipy.sparse.linalg import LinearOperator


def identity(height):
    lin_op = LinearOperator(
            dtype=np.float64,
            matvec=lambda x: x,
            matmat=lambda X: X,
            rmatvec=lambda x: x,
            rmatmat=lambda X: X,
            shape=(height, height),
    )
    return lin_op


class Dataset(BaseDataset):

    name = "Denoising"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # A * I + bruit ~ N(mu, sigma)
    parameters = {
        'std_noise': [0.3],
        'subsampling': [4]}

    def __init__(self, std_noise=0.3, subsampling=4, random_state=27):
        # Store the parameters of the dataset
        self.std_noise = std_noise
        self.subsampling = subsampling
        self.random_state = random_state

    def set_lin_op(self, height):
        return identity(height)

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        img = misc.face(gray=True)[::self.subsampling, ::self.subsampling]
        height, width = img.shape
        lin_op = self.set_lin_op(height)
        y_degraded = (lin_op @ img +
                      rng.normal(0, self.std_noise, size=(height, width)))
        data = dict(lin_op=lin_op, y=y_degraded)

        return data
