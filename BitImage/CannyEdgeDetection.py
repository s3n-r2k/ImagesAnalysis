import numpy as np 
import math

class cannyEdge:
    """
    Written by Simon Rask
    Source: https://en.wikipedia.org/wiki/Canny_edge_detector

    * Apply Gaussian filter to smooth the image in order to remove the noise
    * Find the intensity gradients of the image 
        * We use the Sobel operator
        * Source: https://en.wikipedia.org/wiki/Sobel_operator
    * Apply non-maximum suppression to get rid of spurious response to edge detection
    * Apply double threshold to determine potential edges
    * Track edge by hysteresis: 
       Finalize the detection of edges by suppressing all the other edges that are weak
       and not connected to strong edges.
    """
    def __init__(self):
        return
        
    def GaussFilter(self, kernel_size = 3, sigma = 1.4):
        # Error Checking - kernel dimensions must be odd
        if (kernel_size%2==0):
            return -2

        # Generate kernel
        kernel = np.zeros((kernel_size, kernel_size))

        k = (kernel_size-1)//2
        for i in range(kernel_size):
            for j in range(kernel_size):
                top = (i-(k+1))**2+(j-(k+1))**2
                bott = 2*sigma**2
                kernel[i,j] = np.exp(-top/bott)
        kernel *= 1/(2*math.pi*sigma**2)
        return kernel

    def Sobel_operator(self, img):
        kernel_x = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ])

        kernel_y = np.array([
            [ 1,  2,  1],
            [ 0,  0,  0],
            [-1, -2, -1],
        ])
        G_x = self.conv(img, kernel_x)
        G_y = self.conv(img, kernel_y)

        G1 = np.sqrt(G_x**2 + G_y**2)
        G2 = G_x**2 + G_y**2
        return G1, G2 # TODO there is an major issue here!!


    def conv(self, img, kernel):
        """
        Function that convoludes the image as follows
        img[i,j] = sum_m^n b[m,n]*img[i+m,j+n]
        """
        # Error Checking - inputs
        if not all([isinstance(img, np.ndarray), 
                    isinstance(kernel, np.ndarray)]):
            return -1
   
        # Dimensions of the original image
        m = img.shape[0]
        n = img.shape[1]

        # Error Checking - kernel must be square
        if kernel.shape[0]!=kernel.shape[1]:
            return -2

        # Set kernel size
        kernel_size = kernel.shape[0]
        k = (kernel_size-1)//2

        # Error Checking - kernel dimensions must be odd
        if (kernel_size%2==0):
            return -2

        # Zero-padding the image
        img_padded = np.zeros((m+kernel_size-1,n+kernel_size-1))
        img_padded[k:-k,k:-k] = img
        img_padded[0:k, 0:k] = img[0,0]
        img_padded[0:k, -k:] = img[0,-1]
        img_padded[-k:, 0:k] = img[-1,0]
        img_padded[-k:, -k:] = img[-1,-1]
        for i in range(k):
            img_padded[k:-k, i]  = img[:,0]
            img_padded[k:-k, -i-1] = img[:,-1]
            img_padded[i, k:-k]  = img[0, :]
            img_padded[-i-1, k:-k] = img[-1, :]

        # Initialize output image
        out_img = np.zeros_like(img)

        # Image convolution
        for i in range(m):
            for j in range(n):
               out_img[i,j] = (kernel*img_padded[i:i+kernel_size,j:j+kernel_size]).sum()

        return out_img
