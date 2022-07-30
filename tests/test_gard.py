import pytest
import numpy as np
from scipy.optimize import check_grad as check_grad
from utils.shared import huber
from utils.shared import grad_huber
from utils.shared import make_blur
from benchopt.utils.safe_import import set_benchmark

# this means this test has to be run from the root
set_benchmark('.')

from solvers.ADMM import loss, jac_loss  # noqa: E402


@pytest.mark.parametrize('random_state', [0, 27, 42, 66])
def test_grad_ADMM(random_state):
    n = 10
    m = 10
    rng = np.random.RandomState(random_state)
    A = make_blur('deblurring', n)
    y = rng.normal(0, 1, (n, m))
    delta = 0.9
    zh = rng.randn(n, m)
    zv = rng.randn(n, m)
    muh = rng.randn(n, m)
    muv = rng.randn(n, m)
    gamma = 0.1

    def func(u):
        return loss(y, A, u, delta, n, m, zh, zv, muh, muv, gamma)

    def jac(u):
        return jac_loss(y, A, u, delta, n, m, zh, zv, muh, muv, gamma)

    np.testing.assert_almost_equal(0, check_grad(func, jac,
                                                 x0=rng.randn(n*m)))


def test_grad_huber(random_state):
    n = 10
    m = 10
    rng = np.random.RandomState(random_state)
    delta = 0.9

    def func(y):
        return huber(y, delta)

    def jac(y):
        return grad_huber(y, delta)

    np.testing.assert_almost_equal(0, check_grad(func, jac,
                                                 x0=rng.randn(n, m)))
