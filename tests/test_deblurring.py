import numpy as np

from utils.shared import make_blur


def test_blur_operator_adjoint():
    rng = np.random.RandomState(42)
    u = rng.normal(0, 1., size=(256, 128))
    v = rng.normal(0, 1., size=(256, 128))
    blur = make_blur('deblurring', 256, 27, 3.)
    first_ip = np.sum(np.diag((blur @ u).T @ v))
    second_ip = np.sum(np.diag(u.T @ (blur @ v)))
    np.testing.assert_almost_equal(first_ip, second_ip)


def test_blur_operator_norm():
    rng = np.random.RandomState(42)
    q = rng.normal(0, 1., size=(256, 128))
    blur = make_blur('deblurring', 256, 27, 3.)
    # Power method
    for _ in range(100):
        z = blur @ q
        q = z / np.sqrt(np.sum(z*z))
        mu = np.sum(np.diag(q.T @ (blur @ q)))
        assert mu <= 1.0
