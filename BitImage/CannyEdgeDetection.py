import numpy as np 

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

    def conv(self,img, kernel):
        """
        Function that convoludes the image as follows
        img[i,j] = sum_m^n b[m,n]*img[i+m,j+n]
        """

        pass
