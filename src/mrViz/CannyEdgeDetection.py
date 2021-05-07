import math

import matplotlib.pyplot as plt
import numpy as np


class cannyEdge:
    """
    Written by Simon Rask
    Source: https://en.wikipedia.org/wiki/Canny_edge_detector

    * Apply Gaussian filter to smooth the image in order to remove the noise
    * Find the intensity gradients of the image
        * We use the Sobel operator
        * Source: https://en.wikipedia.org/wiki/Sobel_operator
    * Apply non-maximum suppression
        We apply a non-maximum suppression to get rid of spurious
        response to edge detection
    * Apply double threshold to determine potential edges
    * Track edge by hysteresis:
        Finalize the detection of edges by suppressing all the other
        edges that are weakand not connected to strong edges.
    """

    def __init__(self):
        """
        TODO: Consider if there are some stuff that are nice to initialize
        """
        return

    def GaussFilter(
        self, kernel_size: int = 3, sigma: float = 1.4
    ) -> np.ndarray:
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
        # Error Checking - kernel dimensions must be odd
        if kernel_size % 2 == 0:
            return -2

        # Initialize kernel
        kernel = np.zeros((kernel_size, kernel_size))

        k = (kernel_size - 1) // 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                top = (i - (k + 1)) ** 2 + (j - (k + 1)) ** 2
                bott = 2 * sigma ** 2
                kernel[i, j] = np.exp(-top / bott)
        kernel *= 1 / (2 * math.pi * sigma ** 2)

        return kernel

    def Sobel_operator(
        self, img: np.ndarray, show: bool = False
    ) -> np.ndarray:
        """[summary]

        Parameters
        ----------
        img : np.ndarray
            The image which to apply the Sobel operator
        show : bool, optional
            if True, the result of the operator is displayed, by default Fasle

        Returns
        -------
        np.ndarray
            The sobel operator applied to the image.
        """
        kernel_x = np.array(
            [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1],
            ]
        )

        kernel_y = np.array(
            [
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1],
            ]
        )
        G_x = self.conv(img, kernel_x)

        # If show is True then show the image
        if show:
            plt.imshow(G_x, cmap="gray")
            plt.title("Horizontal Edge")
            plt.show()

        G_y = self.conv(img, kernel_y)
        # If show is True then show the image
        if show:
            plt.imshow(G_y, cmap="gray")
            plt.title("vertical Edge")
            plt.show()

        G_return = np.zeros(img.shape)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                G_return[i, j] = (G_x[i, j] ** 2 + G_y[i, j] ** 2) ** (1 / 2)

        return G_return

    def conv(self, img, kernel, Extend=False):
        """
        Function that convoludes the image as follows
        img[i,j] = sum_m^n b[m,n]*img[i+m,j+n]
        """
        # Error Checking - inputs
        if not all(
            [isinstance(img, np.ndarray), isinstance(kernel, np.ndarray)]
        ):
            return -1

        # Dimensions of the original image
        h_img = img.shape[0]
        w_img = img.shape[1]

        # Error Checking - kernel must be square
        if kernel.shape[0] != kernel.shape[1]:
            return -2

        # Set kernel size
        kernel_size = kernel.shape[0]
        k = (kernel_size - 1) // 2

        # Error Checking - kernel dimensions must be odd
        if kernel_size % 2 == 0:
            return -2

        # Zero-padding the image
        img_padded = np.zeros(
            (h_img + kernel_size - 1, w_img + kernel_size - 1)
        )
        img_padded[k:-k, k:-k] = img

        # Initialize output image
        out_img = np.zeros_like(img)

        # Image convolution
        for row in range(h_img):
            for col in range(w_img):
                for i in range(kernel_size):
                    for j in range(kernel_size):
                        out_img[row, col] += (
                            img_padded[row + i - k, col + j - k] * kernel[i, j]
                        )  # TODO: GO FIX!

        return out_img
