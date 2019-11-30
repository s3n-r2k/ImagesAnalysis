import cv2
import numpy as np 

# Reading an image in default mode 
img = cv2.imread('GoJul.jpg',0)

img_shape = img.shape
h = img_shape[0]
w = img_shape[1]


state = False
for i in range(h):
    for j in range(w):
        if img[i,j]!=255 and not state:
            state = True
            px = i
            py = j

print(px,py, img[px,py] )

