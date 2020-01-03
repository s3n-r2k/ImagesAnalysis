import cv2 
import numpy as np 
from CannyEdgeDetection import cannyEdge
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Reading an image in default mode 
img = cv2.imread('chek.jpg', 0)

img_shape = img.shape

cd = cannyEdge()

ker = np.array([
    [1,0,0],
    [0,0,0],
    [0,0,1]])
res = cd.conv(img, ker)
"""
gauss =  cd.GaussFilter(3,sigma=2)
res = cd.conv(img, gauss)

edge = cd.Sobel_operator(res)
"""
plt.subplot(2,1,1)
plt.imshow(res, cmap='gray')
#plt.title("vertical Edge")

plt.subplot(2,1,2)
plt.imshow(img, cmap='gray')

plt.show()

