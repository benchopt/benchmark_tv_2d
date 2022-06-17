import numpy as np


def get_l2norm(A, n_iter=100):
    # multiplication for the smaller size of matrice
    if A.shape[0] < A.shape[1]:
        A = A.T
    AtA = A.T @ A
    x = np.random.randn(A.shape[1], A.shape[1])
    for _ in range(n_iter):
        x = AtA @ x
        x /= np.linalg.norm(x)
    return np.sqrt(np.linalg.norm(AtA @ x))
