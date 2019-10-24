#6/10 Extract Data Region


import cv2
from matplotlib import pyplot as plt
from easygui import fileopenbox
import numpy as np
origin=cv2.imread(fileopenbox())
origin2=origin.copy()
#origin=cv2.imread("1.jpg")
gray=cv2.cvtColor(origin,cv2.COLOR_RGB2GRAY)
#gray=cv2.imread("4.jpg",0)

ret,img = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)

[a,b]=np.shape(img)#a=273 #b=654

#每一行or列 像素累加值統計
list=[]

#統計每一行
for i in range(0,b):
    count=0
    for j in range(0,a):
        if img[j,i]==0:
            count=count+1
    list.append(count)

plt.figure(1)
plt.plot(list)
plt.show()

#從最左邊選擇一個最大值
for i in range(0,b):
    if abs(list[i]-max(list))<10:
        target=i
        break

show2=origin.copy()
x_label_location=target
print(x_label_location)
#畫X軸
cv2.line(origin, (target, 0), (target, b), (0, 0, 255), 5)
show2=show2[:,target:b]
#從最右邊選擇一個最大值
for i in range(b-1,0,-1):
    if abs(list[i]-max(list))<20:
        target1=i
        break
# Check
if target1>0.9*b:
    #print(target1)
    cv2.line(origin, (target1, 0), (target1, b), (0, 0, 255), 5)
    show2=show2[:,0:target1-target]
else:
    print("找不到右邊界")
list=[]
for i in range(0,a):
    count=0
    for j in range(0,b):
        if img[i,j]==0:
            count=count+1
    list.append(count)
plt.figure(2)
plt.plot(list)
#plt.show()

for i in range(a,0,-1):
    if abs(list[i-1]-max(list))<20:
        target=i
        break
#cv2.imshow("1",origin[0:target,b:target])

y_label_location=target
print(y_label_location)
cv2.line(origin, (0,target), (b,target), (0, 0, 255), 5)
show2=show2[0:target,:]

for i in range(0,a):
    if abs(list[i]-max(list))<20:
        target1=i
        break

#Check
if target1<0.2*a:
    #print(target1)
    cv2.line(origin, (0, target1), (b, target1), (0, 0, 255), 5)
    show2 = show2[target1:target, :]
else:
    print("找不到上邊界")

x_label=origin2[:,0:x_label_location]
y_label=origin2[y_label_location:a,:]
cv2.imshow("DR",show2)
cv2.imwrite("DR.jpg",show2)
cv2.imshow("",origin)
cv2.imshow("x_label",x_label)
cv2.imwrite("x_label.jpg",x_label)
cv2.imshow("y_label",y_label)
cv2.imwrite("y_label.jpg",y_label)
cv2.waitKey()

# 去網格
'''
#去網格

import cv2
import numpy
import sys

BLOCK_SIZE = 50
THRESHOLD = 25


def preprocess(image):
    image = cv2.medianBlur(image, 3)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return 255 - image


def postprocess(image):
    image = cv2.medianBlur(image, 5)
    # image = cv2.medianBlur(image, 5)
    # kernel = numpy.ones((3,3), numpy.uint8)
    # image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return image


def get_block_index(image_shape, yx, block_size):
    y = numpy.arange(max(0, yx[0]-block_size), min(image_shape[0], yx[0]+block_size))
    x = numpy.arange(max(0, yx[1]-block_size), min(image_shape[1], yx[1]+block_size))
    return numpy.meshgrid(y, x)


def adaptive_median_threshold(img_in):
    med = numpy.median(img_in)
    img_out = numpy.zeros_like(img_in)
    img_out[img_in - med < THRESHOLD] = 255
    return img_out


def block_image_process(image, block_size):
    out_image = numpy.zeros_like(image)
    for row in range(0, image.shape[0], block_size):
        for col in range(0, image.shape[1], block_size):
            idx = (row, col)
            block_idx = get_block_index(image.shape, idx, block_size)
            out_image[block_idx] = adaptive_median_threshold(image[block_idx])

    return out_image


def process_image_file(filename):
    image_in = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)

    image_in = preprocess(image_in)
    image_out = block_image_process(image_in, BLOCK_SIZE)
    image_out = postprocess(image_out)

    cv2.imwrite('bin_' + filename, image_out)
if __name__ == "__main__":
    sys.argv[0]=["C:\\Users\\Burny\\PycharmProjects\\remake\\venv"]
    process_image_file("4_data_region.jpg")   
'''

# OCR
'''

from PIL import Image
import pytesseract
import argparse
import cv2
import os
from easygui import fileopenbox


# load the example image and convert it to grayscale

image = cv2.imread(fileopenbox())
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#gray = cv2.medianBlur(gray,3)
#ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
#gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
ret, gray = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

#gray = cv2.medianBlur(gray,3)
#filename = "{}.png".format(os.getpid())
#cv2.imwrite(filename, gray)

# load
# the image as a PIL/Pillow image, apply OCR, and then delete
# the temporary file
#text = pytesseract.image_to_string(image,config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789')
text = pytesseract.image_to_string(gray,lang='engB',config='--psm 6 --oem 1 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
text = pytesseract.image_to_string(gray,lang='engB',config='--psm 6 --oem 1')
#text = pytesseract.image_to_string(gray)
#排列 準備定位
out1 = pytesseract.image_to_data(gray,lang='engB',config='--psm 6 --oem 1 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ' ,output_type=pytesseract.Output.DICT )
out1 = pytesseract.image_to_data(gray,lang='engB',config='--psm 6 --oem 1',output_type=pytesseract.Output.DICT)
#out1 = pytesseract.image_to_data(image,config='--psm 6' ,output_type=pytesseract.Output.DICT )
#out1 = pytesseract.image_to_data(gray,output_type=pytesseract.Output.DICT)
num_boxes = len(out1['level'])
for i in range(num_boxes):
    if out1['text'][i]=="":
        pass
    else:
        (x, y, w, h) = (out1['left'][i], out1['top'][i], out1['width'][i], out1['height'][i])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
print(text)


cv2.imshow("Image", image)
cv2.imshow("Output", gray)

# show the output images
cv2.waitKey(0)

'''
