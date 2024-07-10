import torch
import numpy as np
from scipy.signal import fftconvolve
from scipy.signal.windows import gaussian
from scipy.sparse.linalg import LinearOperator


def get_l2norm(A, n_iter=100):
    """Power method iterations"""
    x = np.random.randn(A.shape[1])
    for _ in range(n_iter):
        x = A.T @ (A @ x)
        norm_x = np.linalg.norm(x)
        x /= norm_x
    return norm_x


def make_blur(type_A, height, size=27, std=8, mask=None):
    if type_A == 'denoising':
        A = LinearOperator(
            dtype=np.float64,
            matvec=lambda x: x,
            matmat=lambda X: X,
            rmatvec=lambda x: x,
            rmatmat=lambda X: X,
            shape=(height, height),
        )
    elif type_A == 'deblurring':
        filt = np.outer(
            gaussian(size, std),
            gaussian(size, std))
        filt /= filt.sum()
        A = LinearOperator(
            dtype=np.float64,
            matvec=lambda x: fftconvolve(x, filt, mode='same'),
            matmat=lambda X: fftconvolve(X, filt, mode='same'),
            rmatvec=lambda x: fftconvolve(x, filt, mode='same'),
            rmatmat=lambda X: fftconvolve(X, filt, mode='same'),
            shape=(height, height),
        )

    elif type_A == 'inpainting':
        if mask is None:
            raise ValueError("missing mask for inpainting.")

        def inpaint(x):
            x = x.reshape((int(np.sqrt(height)), int(np.sqrt(height))))
            x_inpainted = x.copy()
            missing = np.where(mask == 0)
            for i, j in zip(*missing):
                neighbors = []
                if i > 0:
                    neighbors.append(x[i-1, j])
                if i < x.shape[0] - 1:
                    neighbors.append(x[i+1, j])
                if j > 0:
                    neighbors.append(x[i, j-1])
                if j < x.shape[1] - 1:
                    neighbors.append(x[i, j+1])
                if neighbors:
                    x_inpainted[i, j] = np.mean(neighbors)
            return x_inpainted.flatten()

        A = LinearOperator(
            dtype=np.float64,
            matvec=inpaint,
            matmat=lambda X: np.apply_along_axis(inpaint, 0, X),
            rmatvec=inpaint,
            rmatmat=lambda X: np.apply_along_axis(inpaint, 0, X),
            shape=(height, height),
        )
    return A


def huber(R, delta):
    norm_1 = np.abs(R)
    loss = np.where(norm_1 < delta,
                    0.5 * norm_1**2,
                    delta * norm_1 - 0.5 * delta**2)
    return np.sum(loss)


def grad_huber(R, delta):
    return np.where(np.abs(R) < delta, R, np.sign(R) * delta)


def grad_F(y, A, u, data_fit, delta):
    R = A @ u - y
    if data_fit == 'lsq':
        return A.T @ R
    elif data_fit == 'huber':
        return A.T @ grad_huber(R, delta)


def rgb_to_grayscale(rgb_tensor):
    """
    Convert an RGB tensor to a grayscale tensor.

    Parameters:
    rgb_tensor (torch.Tensor): Input tensor of shape (3, n, m).

    Returns:
    torch.Tensor: Grayscale tensor of shape (1, n, m).
    """
    if rgb_tensor.shape[0] != 3:
        raise ValueError("Input tensor must have shape (3, n, m)")

    transform = torch.tensor([0.2989, 0.5870, 0.1140],
                             device=rgb_tensor.device)

    grayscale_tensor = torch.tensordot(rgb_tensor.permute(1, 2, 0),
                                       transform, dims=1)

    grayscale_tensor = grayscale_tensor.unsqueeze(0)

    return grayscale_tensor
