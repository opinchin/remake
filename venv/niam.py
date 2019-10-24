import cv2
import numpy as np
from easygui import fileopenbox
import sys
#from PIL import Image
#from matplotlib import pyplot as plt


def create_mask(col, row):
    mask = np.zeros((col, row))
    cv2.circle(mask, (5, 5), 5, (255, 255, 255), -1, 8, 0)
    return mask


def line_detect(img, l):
    #img = cv2.medianBlur(img, 3)
    edges = cv2.Canny(img, 50, 150)
    lines = cv2.HoughLinesP(edges, 0.1, np.pi/1000, 25, 25, 0)
    print(l)
    lin_num = np.size(lines, 0)
    print(lin_num)
    for i in range(lin_num):
        for x1, y1, x2, y2 in lines[i]:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 3)
  #  kernel = np.ones((2, 2), np.uint8)
    #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
   # img = cv2.dilate(img, kernel, 0)
    return img


def cal_each_x_accumulation(img):
    x_label = []
    list = []
    total = 0
    [row, col] = np.shape(img)
    for i in range(col):
        a = img[:, i]
        for j in range(row):
            count = a[j]
            if count == 255:
                count = 0
            else:
                count = 1
            total += count
        list.append(total)
        x_label.append(i)
        total = 0
    return list, x_label


def cal_each_y_accumulation(img):
    x_label = []
    list = []
    total = 0
    [row, col] = np.shape(img)
    for i in range(row):
        a = img[i, :]
        for j in range(col):
            count = a[j]
            if count == 255:
                count = 0
            else:
                count = 1
            total += count
        list.append(total)
        x_label.append(i)
        total = 0
    return list, x_label


def find_bound(list, startline):
    startline = startline+1
    [bound] = np.shape(list)
    for i in range(startline, bound):
        if list[i] != 0:
            bound_1 = i
            break
        if i == bound-1:
            return None, None
    for i in range(bound_1, bound):
        if list[i] == 0:
            bound_2 = i
            break
    print(bound_1)
    print(bound_2)
    return bound_1, bound_2


def find_bound_inv(list, startline):
    startline = startline-1
    [bound] = np.shape(list)
    for i in range(startline, 0, -1):
        if list[i] != 0:
            bound_1 = i
            break
        if i == 0:
            return None, None
    for i in range(bound_1, 0, -1):
        if list[i] == 0:
            bound_2 = i
            break
    print(bound_1)
    print(bound_2)
    return bound_1, bound_2


def find_roi_bound(roi):
    [roi_list, x] = cal_each_x_accumulation(roi)
    x1 = 0
    roi_x1_list = []
    roi_x2_list = []
    while x1 != None:
        [x1, x2] = find_bound(roi_list, x1)
        roi_x1_list.append(x1)
        roi_x2_list.append(x2)
        x1 = x2

    return roi_x1_list, roi_x2_list


image = cv2.imread(fileopenbox())
hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
kernel = np.uint8([[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, gray = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
eroded = cv2.erode(gray, kernel)
[img_row, img_col, img_channel] = np.shape(image)
#找尋X標籤軸邊界
a = round(img_row/2)
b = round(img_col/2)
x_eroded = eroded[a:img_row, b:img_col]
[a, b] = np.shape(x_eroded)
list, x = cal_each_y_accumulation(x_eroded)
bound_1, y2 = find_bound_inv(list, a)
y2 = y2+a
bound_1 = bound_1 + a
x_label = eroded[y2:bound_1, :]
list, x = cal_each_x_accumulation(x_label)
[a, b] = np.shape(x_label)
[x2, x_roibound2] = find_bound_inv(list, b)

#找尋Y標籤軸邊界
a = round(img_row/2)
b = round(img_col/2)
y_eroded = eroded[0:a, 0:b]
[a, b] = np.shape(y_eroded)
list, x = cal_each_x_accumulation(y_eroded)
bound_1, x1 = find_bound(list, 0)
y_label = eroded[:, bound_1:x1]
list, x = cal_each_y_accumulation(y_label)
[y1, y_roibound2] = find_bound(list, 0)
#定位資料區
image_data = image[y1:y2,x1:x2,:]
cv2.imwrite('dataregion.bmp', image_data)


mask = create_mask(11, 11)
std = round(np.std(hsv_img[:, :, 2], ddof=1), 0)
i = 0
j = 0
k = 0
l = 0
count = 0
total = 0
[data_row, data_col, x] = np.shape(image_data)
[mask_row, mask_col] = np.shape(mask)
image_data_result = np.zeros((data_row, data_col), dtype=np.uint8)

gray_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
ret, gray__image = cv2.threshold(gray_image, 170, 255, cv2.THRESH_BINARY)

for i in range(data_row):
    for j in range(data_col):
        if gray__image[i, j] == 255:
            image_data[i, j, :] = 255
cv2.imwrite("grayy.bmp", gray__image)
cv2.imwrite('after_dataregion.bmp', image_data)
hsv_img = cv2.cvtColor(image_data, cv2.COLOR_BGR2HSV)
cv2.imshow("result1", image_data)
cv2.waitKey()
cv2.destroyWindow("result1")
std = round(np.std(hsv_img[:, :, 2], ddof=1), 0)
print("Calculating")
for i in range(data_row):
    for j in range(data_col):
        for k in range(mask_row):
            for l in range(mask_col):
                if mask[k, l] == 0:
                    pass
                else:
                    temp_a = 5 - k
                    temp_b = 5 - l
                    if (i-temp_a) < 0 or (j-temp_b) < 0 or (i-temp_a) >= data_row or (j-temp_b) >= data_col:
                        pass
                    else:
                        hsv1 = int(hsv_img[i, j, 2])
                        hsv2 = int(hsv_img[i - temp_a, j - temp_b, 2])
                        result = abs(hsv1 - hsv2)
                        if result <= std:
                            result = 1
                        else:
                            result = 0
                        count += 1
                        total += result
        total /= count
        if total < 0.5:
            total = 0.5 - total
            total *= 255
            image_data_result[i, j] = total
        else:
            image_data_result[i, j] = 0
        total = 0
        count = 0

ret, result_img = cv2.threshold(image_data_result, 0, 255, cv2.THRESH_BINARY)

#result_img = cv2.imread("testimg1.jpg", 1)
#result_img = line_detect(result_img, data_col/10)
#result_img = line_detect(result_img)
cv2.imwrite("line_result.bmp", result_img)
cv2.imshow("result1", result_img)
cv2.waitKey()
cv2.destroyAllWindows()
over_lap_img = image_data
for i in range(data_row):
    for j in range(data_col):
        if result_img[i, j] == 255 and gray_image[i, j] == 0: #當result_img[i,j]是白 且 該點於原圖為黑色
            over_lap_img[i, j, :] = 255
        elif result_img[i, j] == 255 and gray_image[i, j] == 255: #當result_img[i,j]是白 且 該點於原圖為白色
            over_lap_img[i, j, :] = 0
        elif result_img[i, j] == 255 and gray_image[i, j] != 0:  # 當result_img[i,j]是白的 且 該點於原圖不為黑色
            over_lap_img[i, j, :] = image_data[i, j, :]
        else:
            over_lap_img[i, j, :] = 0
over_lap_img = cv2.bilateralFilter(over_lap_img, 9, 75, 75)
cv2.imshow("olap", over_lap_img)
cv2.waitKey()
cv2.imwrite("olap.bmp", over_lap_img)
cv2.destroyAllWindows()


print("Finsh")