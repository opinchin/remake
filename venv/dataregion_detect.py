import cv2
import numpy as np
from matplotlib import pyplot as plt


def dataregion_detect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, gray = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    cv2.imshow("gray",gray)

    [a, b] = np.shape(gray)

    image_data = image.copy()
    # 每一行or列 像素累加值統計
    list = []
    # 統計每一行
    for i in range(0, b):
        count = 0
        for j in range(0, a):
            if gray[j, i] == 0:
                count = count + 1
        list.append(count)


    # 由左至右找尋符合邊界(X軸)
    for i in range(0, b):
        if abs(list[i] - max(list)) < a/30:
            if abs(list[i+1] - max(list)) > a/30:
                target = i+1
                break
    left_bound = target
    print(left_bound)

    # X軸
    image_data = img[:, left_bound:b]
    # 由右至左找尋有無右邊界
    for i in range(b - 1, 0, -1):
        if abs(list[i] - max(list)) < a/30:
            if abs(list[i-1] - max(list)) > a / 30:
                target1 = i
                break

    # Check
    if target1 > 0.8 * b:
        image_data = image_data[:, 0:target1 - target]
        right_bound = target1
    # 統計每一列
    list = []
    for i in range(0, a):
        count = 0
        for j in range(0, b):
            if gray[i, j] == 0:
                count = count + 1
        list.append(count)
    # 由下至上找尋符合邊界(Y軸)
    for i in range(a, 0, -1):
        if abs(list[i - 1] - max(list)) < b/30:
            if abs(list[i - 2] - max(list)) > b / 30:
                target = i
                break
    down_bound = target
    # Y軸
    image_data = image_data[0:target, :]
    # 由上至下找尋有無上邊界
    for i in range(0, a):
        if abs(list[i] - max(list)) < b/30:
            if abs(list[i+1] - max(list)) > b / 30:
                target1 = i
                break

    # Check
    if target1 < 0.9 * a:
        image_data = image_data[target1:target, :]
        up_bound = target1
    else:
        print("找不到上邊界")
    cv2.imshow("DataRegion", image_data)
    cv2.waitKey()
    return up_bound, down_bound, left_bound, right_bound


img = cv2.imread("testpaper4.jpg")
list_ = dataregion_detect(img)

