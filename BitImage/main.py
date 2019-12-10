import cv2
import numpy as np 
from CannyEdgeDetection import cannyEdge
# Reading an image in default mode 
img = cv2.imread('Valve.png', 0)

img_shape = img.shape

cd = cannyEdge()

gauss =  cd.GaussFilter(5)
res = cd.conv(img,gauss)

g1, g2 = cd.Sobel_operator(img)

print(type(g1), type(g2))
#cv2.imwrite('imageConv.jpg',10*res)
#cv2.imwrite('imageConv2.jpg', img_sobel)

