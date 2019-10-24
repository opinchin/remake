import cv2
import numpy as np
import Gui_define
img = cv2.imread("Grid_removed (2).jpg")
[a, b, c] = np.shape(img)  # a=484 b=996,c=3
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel = np.ones((5, 5), np.uint8)
gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
ret, thresh1 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
cv2.imshow("thr",thresh1)
cv2.waitKey()

pos = [] # 紀錄該行的pixel值
for i in range(0, a):  # 每一列
    pos.append(thresh1[i, 99])
for i in range(0, len(pos)):
    print(thresh1[i, 99])


#def find_data_location(list):
 #   for i in list:




