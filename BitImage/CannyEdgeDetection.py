import numpy as np 
import math

class cannyEdge:
    """
    Written by Simon Rask
    Source: https://en.wikipedia.org/wiki/Canny_edge_detector

    * Apply Gaussian filter to smooth the image in order to remove the noise
    * Apply Gaussian filter to smooth the image in order to remove the noise
    * Find the intensity gradients of the image
    * Apply non-maximum suppression to get rid of spurious response to edge detection
    * Apply double threshold to determine potential edges
    * Track edge by hysteresis: 
       Finalize the detection of edges by suppressing all the other edges that are weak
       and not connected to strong edges.
    """
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def GaussFilter(self):
        pass

    def conv(img, kernel_size = 3, sigma = 1.4):
        """
        Function that convoludes the image as follows
        img[i,j] = sum_m^n b[m,n]*img[i+m,j+n]
        """
        # Error Checking - input
        if not isinstance(img, np.ndarray):
            return -1
   
        # Dimensions of the original image
        m = img.shape[0]
        n = img.shape[1]

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

        # Zero-padding the image
        # TODO handle zero padding!!
        img_padded = np.zeros((m+kernel_size-1,n+kernel_size-1))
        img_padded[k:-k, k:-k] = img
        
        out_img = np.zeros_like(img)

        for i in range(m):
            for j in range(n):
               out_img[i,j] = (kernel*img_padded[i:i+kernel_size,j:j+kernel_size]).sum()
        return out_img
