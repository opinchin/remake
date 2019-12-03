import cv2
import numpy as np
import scipy.signal
import pytesseract


def grid_space_detect(image):
    import Gui_define
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    row, col = np.shape(image)
    # Y方向之累加值 預測X方向之網格與否
    a, b = Gui_define.cal_each_y_accumulation(image)
    # X方向之累加值 預測Y方向之網格與否
    c, d = Gui_define.cal_each_x_accumulation(image)

    # 檢查X方向之網格線
    temp = col / 3
    # 過濾過低的累加值
    for i in range(0, row):
        if a[i] < temp:
            a[i] = 0
    # 配合find_peaks，避免忽略邊界峰值
    aa = list(a)
    a.insert(0, 0)
    a.insert(len(a), 0)
    # 找尋峰值
    peaks, _ = scipy.signal.find_peaks(a, height=0)
    peaks = peaks - 1
    # 紀錄各峰值間隔
    dist = []
    for i in range(0, len(peaks) - 1):
        temp = peaks[i + 1] - peaks[i]
        dist.append(temp)
    # 透過間隔判別X方向網格存在與否
    # 統計並分析網格間隔為何
    compare = 0
    check = False
    for i in set(dist):
        temp = []
        temp[:] = [abs(x - i) for x in dist]
        count = 0
        acc = 0
        for j in temp:
            if j < row / 100:
                count = count + 1
                acc = acc + j
        if count <= len(temp) / 3 or count == 1:
            pass
        else:
            if not check:
                compare = acc
                x_grid_space = i
                check = True
            elif acc > compare:
                pass
            else:
                compare = acc
                x_grid_space = i
    if check:
        print("預測X方向網格間隔=", x_grid_space)
    else:
        x_grid_space = None
        print("預測沒有X方向網格")

    # 檢查Y方向之網格線
    temp = row / 3
    # 過濾過低的累加值
    for i in range(0, col):
        if c[i] < temp:
            c[i] = 0
    # 配合find_peaks，避免忽略邊界峰值
    cc = list(c)
    c.insert(0, 0)
    c.insert(len(c), 0)
    # 找尋峰值
    peaks1, _ = scipy.signal.find_peaks(c, height=0)
    peaks1 = peaks1 - 1
    # 紀錄各峰值間隔
    dist = []
    for i in range(0, len(peaks1) - 1):
        temp = peaks1[i + 1] - peaks1[i]
        dist.append(temp)
    # 透過間隔判別Y方向網格存在與否
    # 統計並分析網格間隔為何
    compare = 0
    check = False
    for i in set(dist):
        temp = []
        temp[:] = [abs(x - i) for x in dist]
        count = 0
        acc = 0
        for j in temp:
            if j < col / 100:
                count = count + 1
                acc = acc + j
        if count <= len(temp) / 3 or count == 1:
            pass
        else:
            if not check:
                compare = acc
                y_grid_space = i
                check = True
            elif acc > compare:
                pass
            else:
                compare = acc
                y_grid_space = i
    if check:
        print("預測Y方向網格間隔=", y_grid_space)
    else:
        y_grid_space = None
        print("預測沒有Y方向網格")
    # aa = X方向累加值, cc = Y方向累加值, peaks = X方向累加值峰值
    return x_grid_space, y_grid_space, aa, cc, peaks, peaks1


def dataregion_detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, gray = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    [a, b] = np.shape(gray)
    image_data = img.copy()
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
            target = i
            break
    left_bound = target
    # X軸
    image_data = img[:, target:b]
    # 由右至左找尋有無右邊界
    for i in range(b - 1, 0, -1):
        if abs(list[i] - max(list)) < a/30:
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
            target = i
            break
    down_bound = target
    # Y軸
    image_data = image_data[0:target, :]
    # 由上至下找尋有無上邊界
    for i in range(0, a):
        if abs(list[i] - max(list)) < b/30:
            target1 = i
            break

    # Check
    if target1 < 0.9 * a:
        image_data = image_data[target1:target, :]
        up_bound = target1
    else:
        print("找不到上邊界")
    # cv2.imshow("DataRegion", image_data)
    return up_bound, down_bound, left_bound, right_bound


img = cv2.imread("testpaper4.jpg")
# legend_remove = cv2.imread("Legend Removed")
row, col, _ = np.shape(img)
up_bound, down_bound, left_bound, right_bound = dataregion_detect(img)
# print(x, y)
x_label = img[down_bound:row, :]
y_label = img[:, 0:left_bound]
x_label_fix = x_label[:, left_bound:right_bound]
y_label_fix = y_label[up_bound:down_bound, :]
grid = False
# x_grid_space, y_grid_space, aa, cc, peaks, peaks1 = grid_space_detect(legend_remove)
check_value = []
check_place = []


if grid:
    text = pytesseract.image_to_string(y_label_fix, lang='engB', config='--psm 6 --oem 1')
    out1 = pytesseract.image_to_data(y_label_fix, lang='engB', config='--psm 6 --oem 1',
                                     output_type=pytesseract.Output.DICT)
    num_boxes = len(out1['level'])
    for i in range(0, num_boxes):
        if not out1['text'][i].isdigit():
            pass
        else:
            (x, y, w, h) = (out1['left'][i], out1['top'][i], out1['width'][i], out1['height'][i])
            if abs((y + h / 2) / x_grid_space - round((y + h / 2) / x_grid_space)) < 0.1:
                check_value.append(int(out1['text'][i]))
                check_place.append(int(round((y + h / 2) / x_grid_space)))
                # print("Got Label In (", x, y, ") and the Value is (", out1['text'][i],  ")")
    check = []
    check_place_ = []
    find_or_not = False
    for i in range(0, len(check_value)):
        if find_or_not:
            break
        else:
            check[:] = [abs(x - check_value[i]) for x in check_value]
            check_place_[:] = [x - (i + 1) for x in check_place]
            for j in range(0, len(check_value)):
                try:
                    check[j] = abs(check[j] / check_place_[j])
                except ZeroDivisionError:
                    pass
            for j in set(check):
                if check.count(j) > len(check_value) / 2:
                    print("Label Define")
                    print("")
                    thr_value = check_value[check.index(j)]
                    thr_place = x_grid_space*check_place[check.index(j)]
                    check[check.index(j)] = 0
                    thr1_value = check_value[check.index(j)]
                    thr1_place = x_grid_space*check_place[check.index(j)]
                    find_or_not = True
                    break
    print("Y軸Label參考點一:位於", thr_place, "Value = ", thr_value)
    print("Y軸Label參考點二:位於", thr1_place, "Value = ", thr1_value)

    out1 = pytesseract.image_to_data(x_label_fix, lang='engB', config='--psm 6 --oem 1',
                                     output_type=pytesseract.Output.DICT)
    num_boxes = len(out1['level'])
    for i in range(0, num_boxes):
        if not out1['text'][i].isdigit():
            pass
        else:
            (x, y, w, h) = (out1['left'][i], out1['top'][i], out1['width'][i], out1['height'][i])
            if abs((x + w / 2) / y_grid_space - round((x + w / 2) / y_grid_space)) < 0.1:
                check_value.append(int(out1['text'][i]))
                check_place.append(int(round((x + w / 2) / y_grid_space)))
                # print("Got Label In (", x, y, ") and the Value is (", out1['text'][i], ")")
    check = []
    check_place_ = []
    find_or_not = False
    for i in range(0, len(check_value)):
        if find_or_not:
            break
        else:
            check[:] = [abs(x - check_value[i]) for x in check_value]
            check_place_[:] = [x - (i + 1) for x in check_place]
            for j in range(0, len(check_value)):
                try:
                    check[j] = abs(check[j] / check_place_[j])
                except ZeroDivisionError:
                    pass
            for j in set(check):
                if check.count(j) > len(check_value) / 2:
                    print("Label Define")
                    print("")
                    thr_value = check_value[check.index(j)]
                    thr_place = y_grid_space * check_place[check.index(j)]
                    check[check.index(j)] = 0
                    thr1_value = check_value[check.index(j)]
                    thr1_place = y_grid_space * check_place[check.index(j)]
                    find_or_not = True
                    break
    print("X軸Label參考點一:位於", thr_place, "Value = ", thr_value)
    print("X軸Label參考點二:位於", thr1_place, "Value = ", thr1_value)
else:
    text = pytesseract.image_to_string(y_label_fix, lang='engB', config='--psm 6 --oem 1')
    print(text)
    print("")
    text = pytesseract.image_to_string(x_label_fix, lang='engB', config='--psm 6 --oem 1')
    print(text)
    cv2.imshow("y",y_label_fix)
    cv2.imshow("x",x_label_fix)
    cv2.waitKey()