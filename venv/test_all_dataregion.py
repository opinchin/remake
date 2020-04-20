import os
import cv2
import numpy as np
import pytesseract
from decimal import Decimal


def dataregion_detect(img):
    global upbound, downbound, leftbound, rightbound, image_data, origin
    try:
        # print("img", img)
        origin = cv2.imread(img)
        gray = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
        ret, gray = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
        [a, b] = np.shape(gray)
        image_data = origin.copy()
        print("Ro =", a ,"Co =", b)
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
            # print("找不到右邊界")

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
            # print("找不到上邊界")
        downbound = target
        cv2.imwrite(os.path.join(save_path, str(img)+'data_region.jpg'), image_data)
        # return leftbound, rightbound, downbound, upbound
        [a,b,c]= np.shape(image_data)
        # print("DownBound =",downbound,"LeftBound =",leftbound)
        print("DownBound =", downbound, "UpBound =", upbound)
        print("RightBound =", rightbound, "LeftBound =", leftbound)

        print("Rd =", a, "Cd =", b)

        print("DataRegion[Row, Col] = [", a, b, ']')

    except:
        print("Error in", img)


def label_define_fun(left, right, down, up, img):
    global y_label_place_1, y_label_place_2, y_label_value_1, y_label_value_2
    global x_label_place_1, x_label_place_2, x_label_value_1, x_label_value_2
    origin = cv2.imread(img)
    cv2.imshow("origin", origin)
    [row, col, _] = np.shape(origin)
    print("Origin[Row, Col] = [", row, col, ']')
    x_label = origin[downbound:row, :]
    y_label = origin[:, 0:leftbound]
    x_label_fix = x_label
    y_label_fix = y_label
    x_label_show = x_label_fix.copy()
    y_label_show = y_label_fix.copy()
    [row, col, _] = np.shape(y_label_fix)
    print("Y_Label[Row, Col] = [", row, col, "]")

    [row, col, _] = np.shape(x_label_fix)
    print("X_Label[Row, Col] = [", row, col, "]")
    cv2.imwrite("x_label.jpg",x_label_fix)
    cv2.imwrite("y_label.jpg",y_label_fix)

    # Label_X
    # output = pytesseract.image_to_data(x_label_fix, lang='engB', config='--psm 6 --oem 1',
    #                                    output_type=pytesseract.Output.DICT)

    # Label_Y
    out1 = pytesseract.image_to_data(y_label_fix,  config='--psm 6 --oem 1',
                                     output_type=pytesseract.Output.DICT)
    # print("Output =", out1)
    num_boxes = len(out1['level'])
    check_value = []  # Label's Value
    check_place = []  # Label's Place
    thr_value = None
    thr1_value = None
    thr1_place = None
    thr_place = None

    for i in range(0, num_boxes):
        # (x, y, w, h) = (out1['left'][i], out1['top'][i], out1['width'][i], out1['height'][i])
        # cv2.rectangle(y_label_show, (x, y), (x + w, y + h), (0, 0, 255), 2)
        try:
            # Check OCR's output k is a digit or not.
            k = float(out1['text'][i])
            (x, y, w, h) = (out1['left'][i], out1['top'][i], out1['width'][i], out1['height'][i])
            check_value.append(k)
            check_place.append(round((y + h / 2)))
            cv2.rectangle(y_label_show, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # print(k)
        except ValueError:
            pass
    cv2.imshow("y_label", y_label_show)
    # cv2.waitKey()

    if check_value == [] or len(check_value) <= 1:
        print("無法正確偵測")
    elif len(check_value) == 2:
        print("資料匱乏，啟用絕對定位")
    else:
        find_or_not = False
        for i in check_value:
            if find_or_not:
                break
            check_value_dist = check_value.copy()  # 確認Label's Value間隔
            check_value_dist[:] = [Decimal(str(x)) - Decimal(str(i)) for x in check_value]
            place = check_value.index(i)
            check_place_dist = check_place.copy()
            check_place_dist[:] = [Decimal(str(x)) - Decimal(str(check_place[place])) for x in check_place]


            check = check_place.copy()
            for j in range(0, len(check_value)):
                if check_place_dist[j] == 0:
                    pass
                else:
                    temp = check_value_dist[j]/check_place_dist[j]
                    # check_origin[j] = temp
                    if check_value[0] > 0:
                        temp = round(temp, 2)
                    else:
                        temp = round(temp, 4)
                    check[j] = temp
            # Label_Text = check_value.copy()
            # Label_Place = check_place.copy()
            # Label_Text_dist = check_value_dist.copy()
            # Label_Place_dist = check_place_dist.copy()
            # Label_Check = check_origin.copy()
            # print("Label_Text =", check_origin)
            # print("Label_Place =", check_origin)
            # print("Label_Check =",check_origin)
            # Value/Place後比較各組比例是否一致
            for j in set(check):
                if check.count(j) >= 2:
                    # print("Label Define")
                    find_or_not = True
                    thr_value = i
                    thr_place = check_place[check_value.index(i)]
                    thr1_value = check_value[check.index(j)]
                    thr1_place = check_place[check.index(j)]
                    break
        print("Y1 =", thr_place)
        print("Y2 =", thr1_place)

        if thr_place != None:
            # 修正位置
            if upbound != None:
                if thr_place - upbound < 0:
                    thr_place = 0
                    thr1_place = thr1_place - upbound
                else:
                    thr_place = thr_place - upbound
                    thr1_place = thr1_place - upbound
            if thr1_place > downbound:
                thr1_place = downbound
            print("Yref1 =", thr_place)
            print("Yref2 =", thr1_place)
            y_label_place_1 = thr_place
            y_label_value_1 = thr_value
            y_label_place_2 = thr1_place
            y_label_value_2 = thr1_value
            print("Y軸Label參考點一:位於", y_label_place_1, "Value = ", y_label_value_1)
            print("Y軸Label參考點二:位於", y_label_place_2, "Value = ", y_label_value_2)
        else:
            print("無法正確偵測Label")

    # Label_Y
    out1 = pytesseract.image_to_data(x_label_fix,  config='--psm 6 --oem 1',
                                     output_type=pytesseract.Output.DICT)
    thr_value = None
    thr1_value = None
    thr1_place = None
    thr_place = None
    num_boxes = len(out1['level'])
    check_value = []  # Label's Value
    check_place = []  # Label's Place
    for i in range(0, num_boxes):
        try:
            # Check OCR's output k is a digit or not.
            k = float(out1['text'][i])
            (x, y, w, h) = (out1['left'][i], out1['top'][i], out1['width'][i], out1['height'][i])
            check_value.append(k)
            check_place.append(round((x + w / 2)))
            cv2.rectangle(x_label_show, (x, y), (x + w, y + h), (0, 0, 255), 2)
        except ValueError:
            pass

    if check_value == [] or len(check_value) <= 1:
        print("無法正確偵測")
    elif len(check_value) == 2:
        print("資料匱乏，啟用絕對定位")
    else:
        find_or_not = False
        for i in check_value:
            if find_or_not:
                break
            check_value_dist = check_value.copy()  # 確認Label's Value間隔
            check_value_dist[:] = [Decimal(str(x)) - Decimal(str(i)) for x in check_value]
            place = check_value.index(i)
            check_place_dist = check_place.copy()
            check_place_dist[:] = [Decimal(str(x)) - Decimal(str(check_place[place])) for x in check_place]
            check = check_place.copy()
            for j in range(0, len(check_value)):
                if check_place_dist[j] == 0:
                    pass
                else:
                    temp = check_value_dist[j]/check_place_dist[j]
                    if abs(check_value[1]) > 0:
                        temp = round(temp, 2)
                    else:
                        temp = round(temp, 4)
                    check[j] = temp
            # Value/Place後比較各組比例是否一致
            for j in set(check):
                if check.count(j) >= 2:
                    # print("Label Define")
                    find_or_not = True
                    thr_value = i
                    thr_place = check_place[check_value.index(i)]
                    thr1_value = check_value[check.index(j)]
                    thr1_place = check_place[check.index(j)]
                    break
            print("X1 =", thr_place)
            print("X2 =", thr1_place)
        if thr_place != None:
            # 修正位置
            thr1_place = thr1_place - leftbound
            thr_place = thr_place - leftbound

            if thr_place < 0:
                thr_place = 0
            if rightbound != None:
                if thr1_place > rightbound:
                    thr1_place = rightbound

            x_label_place_1 = thr_place
            x_label_value_1 = thr_value
            x_label_place_2 = thr1_place
            x_label_value_2 = thr1_value
            print("Xref1 =", thr_place)
            print("Xref2 =", thr1_place)
            print("X軸Label參考點一:位於", x_label_place_1, "Value = ", x_label_value_1)
            print("X軸Label參考點二:位於", x_label_place_2, "Value = ", x_label_value_2)
        else:
            print("無法正確偵測LABEL")
        cv2.imshow("x_label", x_label_show)
        cv2.waitKey()






path = 'C:/Users/Burny/PycharmProjects/remake/venv/all_origin'
save_path='C:/Users/Burny/PycharmProjects/remake/venv/all_origin/dataregion'
os.chdir(path)
allfile = os.listdir(path)
name = 100
# all_rename
# for i in allfile:
#     fname = os.path.splitext(i)[0]  # 分解出當前的檔案路徑名字
#     ftype = os.path.splitext(i)[1]  # 分解出當前的副檔名
#     print(ftype)
#     os.rename(i, str(name)+ftype)
#     name = name+1
# mult_test
# for i in allfile:
#     # print(os.getcwd())
#     dataregion_detect(i)
#
#     cv2.imshow("origin", origin)
#     cv2.imshow("dataregion", image_data)
#     cv2.waitKey()
#     label_define_fun(leftbound, rightbound, downbound, upbound, i)
#     cv2.destroyAllWindows()
#     print(i)
# Single Test
i = "1 (15).JPG"
# i = "1 (20).jpg"
dataregion_detect(i)
cv2.imwrite("origin.jpg", origin)
cv2.imwrite("dataregion.jpg", image_data)
label_define_fun(leftbound, rightbound, downbound, upbound, i)
cv2.destroyAllWindows()
print(i)

# single_TEST
# img = cv2.imread("102.jpg")
# img = "2 (3).jpg"
# # text = pytesseract.image_to_string(img[:, 0:40], lang='equ', config='--psm 6 --oem 1')
# dataregion_detect(img)
# label_define_fun(leftbound, rightbound, downbound, upbound, img)
# text = pytesseract.image_to_string(img[:, 0:70], config='--psm 6 --oem 1')
# print(text)
# cv2.imshow("", img[:, 0:70])
# cv2.waitKey()