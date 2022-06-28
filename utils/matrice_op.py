import numpy as np


def grad_F(y, A, u, data_fit, delta):
    R = A @ u - y
    if data_fit == 'lsq':
        return A.T @ R
    elif data_fit == 'huber':
        return A.T @ np.where(np.abs(R) < delta, R, np.sign(R) * delta)


def prox_huber(u, mu, delta):
    return np.where(
        np.abs(u) <= delta * (mu + 1.0),
        u / (mu + 1.0),
        u - delta * mu * np.sign(u),
    )


def div(vh, vv):
    dh = np.vstack(
        (np.diff(vh, prepend=0, axis=0)[:-1, :], -vh[-2, :])
    )
    dv = np.column_stack(
        (np.diff(vv, prepend=0, axis=1)[:, :-1], -vv[:, -2])
    )
    return dh + dv


def grad(u):
    # Neumann condition
    gh = np.pad(np.diff(u, axis=0), ((0, 1), (0, 0)), 'constant')
    gv = np.pad(np.diff(u, axis=1), ((0, 0), (0, 1)), 'constant')
    return gh, gv


def dual_prox_tv_aniso(vh, vv, reg):
    return np.clip(vh, -reg, reg), \
        np.clip(vv, -reg, reg)


def dual_prox_tv_iso(vh, vv, reg):
    norms = np.sqrt(vh ** 2 + vv ** 2)
    factors = 1. / np.maximum(1, 1./reg * norms)
    return vh * factors, vv * factors
