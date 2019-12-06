import cv2
import numpy as np 
from CannyEdgeDetection import cannyEdge as cd
# Reading an image in default mode 
img = cv2.imread('Valve.png', 0)

img_shape = img.shape
res = cd.conv(img, 5)

cv2.imwrite('imageConv.jpg',res)

