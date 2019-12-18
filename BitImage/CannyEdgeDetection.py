import numpy as np 
import matplotlib.pyplot as plt
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

    def Sobel_operator(self, img, show = True):
        kernel_x = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ])

        kernel_y = np.array([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1],
        ])
        G_x = self.conv(img, kernel_x)
        # If show is True then show the image
        if show:
            plt.imshow(G_x, cmap='gray')
            plt.title("Horizontal Edge")
            plt.show()

        G_y = self.conv(img, kernel_y)
        # If show is True then show the image
        if show:
            plt.imshow(G_y, cmap='gray')
            plt.title("vertical Edge")
            plt.show()

        #G_return =  np.sqrt(np.square(G_x) + np.square(G_y))
        #G_return *= 255.0 / G_return.max()
        G_return = np.zeros(img.shape)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                G_return[i,j] = (G_x[i,j]**2+G_y[i,j]**2)**(1/2)
                
        return G_return 


    def conv(self, img, kernel, Extend=False):
        """
        Function that convoludes the image as follows
        img[i,j] = sum_m^n b[m,n]*img[i+m,j+n]
        """
        # Error Checking - inputs
        if not all([isinstance(img, np.ndarray), 
                    isinstance(kernel, np.ndarray)]):
            return -1
        #kernel = np.flipud(np.fliplr(kernel))
        # Dimensions of the original image
        h_img = img.shape[0]
        w_img = img.shape[1]

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
        img_padded = np.zeros((h_img+kernel_size-1,w_img+kernel_size-1))
        img_padded[k:-k,k:-k] = img
       
        # Initialize output image
        out_img = np.zeros_like(img)

        # Image convolution
        for row in range(h_img):
            for col in range(w_img):
                for i in range(kernel_size):
                    for j in range(kernel_size):
                        out_img[row,col]+=img_padded[row+i,col+j]*kernel[i,j]

        return out_img
