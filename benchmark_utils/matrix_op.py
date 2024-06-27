import numpy as np


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
    if u.ndim == 3:  # RGB image
        gh = np.zeros_like(u)
        gv = np.zeros_like(u)
        for c in range(u.shape[0]):
            gh[c] = np.pad(np.diff(u[c], axis=0), ((0, 1), (0, 0)), 'constant')
            gv[c] = np.pad(np.diff(u[c], axis=1), ((0, 0), (0, 1)), 'constant')

    elif u.ndim == 2:
        gh = np.pad(np.diff(u, axis=0), ((0, 1), (0, 0)), 'constant')
        gv = np.pad(np.diff(u, axis=1), ((0, 0), (0, 1)), 'constant')
    else:
        raise ValueError("Input array must have 2 or 3 dimensions.")

    return gh, gv


def dual_prox_tv_aniso(vh, vv, reg):
    return np.clip(vh, -reg, reg), np.clip(vv, -reg, reg)


def dual_prox_tv_iso(vh, vv, reg):
    norms = np.sqrt(vh ** 2 + vv ** 2)
    factors = 1. / np.maximum(1, 1./reg * norms)
    return vh * factors, vv * factors
