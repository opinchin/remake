import os
import cv2
import numpy as np
import pytesseract


def dataregion_detect(img):
    global upbound, downbound, leftbound, rightbound
    try:
        # print("img", img)
        origin = cv2.imread(img)
        gray = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
        ret, gray = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
        [a, b] = np.shape(gray)
        image_data = origin.copy()
        list = []
        for i in range(0, b):
            count = 0
            for j in range(0, a):
                if gray[j, i] == 0:
                    count = count + 1
            list.append(count)
        for i in range(0, b):
            if abs(list[i] - max(list)) < a / 30:
                target = i
                break
        leftbound = target
        # Y軸
        image_data = origin[:, target:b]
        # 由右至左找尋有無右邊界
        for i in range(b - 1, 0, -1):
            if abs(list[i] - max(list)) < a / 30:
                target1 = i
                break
        # Check
        if target1 > 0.8 * b:
            image_data = image_data[:, 0:target1 - target]
            rightbound = target1
        else:
            rightbound = None
            print("找不到右邊界")

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
            if abs(list[i - 1] - max(list)) < b / 30:
                if abs(list[i - 2] - max(list)) > b / 30:
                    target = i - 1
                    break
        # X軸
        image_data = image_data[0:target, :]
        # 由上至下找尋有無上邊界
        for i in range(0, a):
            if abs(list[i] - max(list)) < b / 30:
                if abs(list[i + 1] - max(list)) > b / 30:
                    target1 = i + 1
                    break
        # Check
        if target1 < 0.9 * a:
            image_data = image_data[target1:target, :]
            upbound = target1
        else:
            upbound = None
            print("找不到上邊界")
        downbound = target
        cv2.imwrite(os.path.join(save_path, str(img)+'data_region.jpg'), image_data)
        # return leftbound, rightbound, downbound, upbound
    except:
        print("Error in", img)


def label_define_fun(left, right, down, up, img):
    global y_label_place_1, y_label_place_2, y_label_value_1, y_label_value_2
    global x_label_place_1, x_label_place_2, x_label_value_1, x_label_value_2
    x_label = origin[downbound:row, :]
    y_label = origin[:, 0:leftbound]
    if rightbound != None:
        x_label_fix = x_label[:, leftbound:rightbound]
    else:
        x_label_fix = x_label
    if upbound != None:
        y_label_fix = y_label[upbound:downbound, :]
    else:
        y_label_fix = y_label
    x_label_show = x_label_fix.copy()
    y_label_show = y_label_fix.copy()

    # Label_X
    output = pytesseract.image_to_data(x_label_fix, lang='engB', config='--psm 6 --oem 1',
                                       output_type=pytesseract.Output.DICT)

    # Label_Y
    output = pytesseract.image_to_data(x_label_fix, lang='engB', config='--psm 6 --oem 1',
                                       output_type=pytesseract.Output.DICT)
    num_boxes = len(out1['level'])
    for i in range(0, num_boxes):
        try:
            # Check OCR's output k is a digit or not.
            k = float(out1['text'][i])
            (x, y, w, h) = (out1['left'][i], out1['top'][i], out1['width'][i], out1['height'][i])
            check_value.append(k)
            check_place.append(round((y + h / 2)))
            cv2.rectangle(y_label_show, (x, y), (x + w, y + h), (0, 0, 255), 2)
        except ValueError:
            pass



path = 'C:/Users/Burny/PycharmProjects/remake/venv/all_origin'
save_path='C:/Users/Burny/PycharmProjects/remake/venv/all_origin/dataregion'
# os.chdir(path)
# allfile = os.listdir(path)
name = 100

# for i in allfile:
#     fname = os.path.splitext(i)[0]  # 分解出當前的檔案路徑名字
#     ftype = os.path.splitext(i)[1]  # 分解出當前的副檔名
#     print(ftype)
#     os.rename(i, str(name)+ftype)
#     name = name+1
# for i in allfile:
#     # print(os.getcwd())
#     left, right, down, up = dataregion_detect(i)
#     print(i)

# img = cv2.imread("1.png")
#
# # text = pytesseract.image_to_string(img[:, 0:40], lang='equ', config='--psm 6 --oem 1')
# # text = pytesseract.image_to_string(img[:, 0:40], config='--psm 6 --oem 1')
# print(text)
# cv2.imshow("", img[:, 0:40])
# cv2.waitKey()