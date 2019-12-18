import cv2 
import numpy as np 
from CannyEdgeDetection import cannyEdge
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Reading an image in default mode 
img = cv2.imread('Valve.png', 0)

img_shape = img.shape

cd = cannyEdge()

gauss =  cd.GaussFilter(3)
res = cd.conv(img, gauss)

edge = cd.Sobel_operator(res)

plt.imshow(edge, cmap='gray')
#plt.title("vertical Edge")
plt.show()

