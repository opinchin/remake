import cv2
import numpy as np
img = cv2.imread("Grid_removed (2).jpg")
[a, b, c] = np.shape(img)  # a=420 b=959,c=3
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel = np.ones((5, 5), np.uint8)
gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
ret, thresh1 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

