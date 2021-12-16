from typing import Tuple

import numba as nb
import numpy as np
from numba import njit, prange
from numba.core.types.containers import UniTuple
from numba.core.types.npytypes import Array

# Set numba image type
ImageType = Array(nb.float64, 2, "C")


@njit((ImageType)(nb.int64, nb.float64))
def GaussFilter(kernel_size: int = 3, sigma: float = 1.4) -> np.ndarray:
    """Gaussian Filter

    Method that creates Gaussian filter

    Parameters
    ----------
    kernel_size : int, optional
        The kernel size, must be odd, by default 3
    sigma : float, optional
        The sigma value, determines the blur, by default 1.4

    Returns
    -------
    np.ndarray
        The gaussian filter, a kernel_size x kernel_size matrix.
    """

    # Initialize kernel
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)

    k = (kernel_size - 1) // 2
    for i in range(1, kernel_size + 1):
        for j in range(1, kernel_size + 1):
            top = (i - (k + 1)) ** 2 + (j - (k + 1)) ** 2
            bott = 2 * sigma ** 2
            kernel[i - 1, j - 1] = np.exp(-top / bott)
    kernel *= 1 / (2 * np.pi * sigma ** 2)

    return kernel


@njit((ImageType)(ImageType, ImageType), parallel=True)
def conv(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Image convolution

    Preforms the convolution:
        output = img * kernel, where * denotes convolution
    and
        output[i, j] = sum_m^n kernel[m,n]*img[i+m,j+n]

    Parameters
    ----------
    img : np.ndarray
        An image
    kernel : np.ndarray
        convolution kernel

    Returns
    -------
    np.ndarray
        convoluded image
    """

    # Dimensions of the original image
    h_img, w_img = img.shape

    # Set kernel size
    kernel_size, _ = kernel.shape
    k = (kernel_size - 1) // 2

    # Zero-padding the image
    img_padded = np.zeros((h_img + kernel_size - 1, w_img + kernel_size - 1))
    img_padded[k:-k, k:-k] = img

    # Initialize output image
    out_img = np.zeros_like(img)

    # Image convolution
    # here using numba.prange in order to execute in parallel
    for row in prange(h_img):
        for col in prange(w_img):
            for i in prange(kernel_size):
                for j in prange(kernel_size):
                    out_img[row, col] += (
                        img_padded[row + i, col + j] * kernel[i, j]
                    )
    return out_img


@njit((UniTuple(ImageType, 2))(ImageType))
def Sobel_operator(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sobel Operator

    Applies the Sobel operator to an image.

    Parameters
    ----------
    img : np.ndarray
        The image which to apply the Sobel operator

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The Gradient magnitude of the image, and the Gradient direction
    """

    kernel_x = np.array(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ],
        dtype=np.float64,
    )

    kernel_y = np.array(
        [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
        ],
        dtype=np.float64,
    )
    G_x = conv(img, kernel_x)
    G_y = conv(img, kernel_y)
    G_return = np.zeros(img.shape)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            G_return[i, j] = (G_x[i, j] ** 2 + G_y[i, j] ** 2) ** (1 / 2)

    # G_return = G_return / G_return.max() * 255
    theta = np.arctan2(G_y, G_x)

    return G_return, theta


@njit((ImageType)(ImageType), parallel=True)
def reflect_angle(T):
    N, M = T.shape
    for i in prange(N):
        for j in prange(M):
            if T[i, j] < 0:
                T[i, j] += np.pi
    return T


@njit((ImageType)(ImageType, ImageType))
def non_max_suppression(G, theta):
    # Reflect angles
    theta = reflect_angle(theta)

    M, N = G.shape
    Z = np.zeros((M, N), dtype=np.float64)

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            g0 = 255
            g1 = 255
            angle = theta[i, j]
            if (0 <= angle < np.pi / 8) or (7 * np.pi / 8 <= angle):
                g0 = G[i, j + 1]
                g1 = G[i, j - 1]
            elif np.pi / 8 <= angle < 3 * np.pi / 8:
                g0 = G[i + 1, j - 1]
                g1 = G[i - 1, j + 1]
            elif 3 * np.pi / 8 <= angle < 5 * np.pi / 8:
                g0 = G[i + 1, j]
                g1 = G[i - 1, j]
            elif 5 * np.pi / 8 <= angle < 7 * np.pi / 8:
                g0 = G[i - 1, j - 1]
                g1 = G[i + 1, j + 1]

            if (G[i, j] >= g0) and (G[i, j] >= g1):
                Z[i, j] = G[i, j]

    # Z = Z / Z.max() * 255
    return Z
