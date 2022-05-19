import numpy as np
import pytest

from datasets.deblurring import make_blur


def test_blur_operator_adjoint():
    rng = np.random.RandomState(42)
    u = rng.normal(0, 1., size=(256,128))
    v = rng.normal(0, 1., size=(256,128))
    blur = make_blur(27, 3.)
    first_ip = np.sum(np.diag(blur(u).T @ v))
    second_ip = np.sum(np.diag(u.T @ blur(v)))
    np.testing.assert_almost_equal(first_ip, second_ip)
