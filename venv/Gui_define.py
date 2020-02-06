import tkinter
import cv2
import PIL.Image, PIL.ImageTk
from easygui import fileopenbox
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema
import pytesseract
import scipy.signal


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
    if startline >= bound:
        return None, None
    for i in range(startline, bound):
        if list[i] != 0:
            bound_1 = i
            break
        if i == bound-1:
            # return
            return None, None
    for i in range(bound_1, bound):
        if i == bound-1 and list[i] == 255:
            bound_2 = i
            break
        if list[i] == 0:
            bound_2 = i
            break
    # print(bound_1)
    # print(bound_2)
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


# 輸入一個list 查找所有數值的的區域(0012342100) 回傳邊界序列
def find_total_bound(list):
    k1 = 0
    k1_list = []
    k2_list = []
    k3_list = []
    k4_list = []
    while k1 <= len(list):
        if k1 == len(list)-1:
            break
        [k1, k2] = find_bound(list, k1)
        try:
            if abs(k1 - k2) > len(list)/2:
                break
        except:
            pass
        if k1 == k2 and k1 == len(list)-1:
            k1_list.append(k1-1)
            k2_list.append(k2)
            temp = 1
            # 邊界寬度
            k3_list.append(temp)
            # 平均累加值
            k4_list.append(sum(list[k1:k2]) / temp)
            break

        if k1 == None:
            break
        else:
            k1_list.append(k1)
            k2_list.append(k2)
            temp = k2 - k1
            # 邊界寬度
            k3_list.append(temp)
            # 平均累加值
            k4_list.append(sum(list[k1:k2]) / temp)
            k1 = k2
    return k1_list, k2_list , k3_list , k4_list


# detect legend
def legend_locate(img):
    rows, cols, channels = img.shape
    mask = np.zeros([rows, cols, 3], dtype=np.uint8)
    origin = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    _, threshold = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(threshold, -1, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, False)
            # 確認是否為正立四邊形且不能為最外圍之邊框
            if len(approx) == 4:
                if abs(approx[0][0][0] - approx[1][0][0]) < 2 and abs(
                        approx[2][0][0] - approx[3][0][0] < 2 and abs(approx[0][0][1] - approx[3][0][1]) < 2
                        and abs(approx[1][0][1] - approx[2][0][1]) < 2 and not abs(
                            approx[0][0][0] - approx[2][0][0]) > 0.8 * cols):
                    # 確認是否有非圖例之誤偵測
                    temp = threshold[approx[0][0][1]:approx[1][0][1], approx[0][0][0]:approx[2][0][0]]
                    [a, b] = np.shape(temp)
                    count = 0
                    countall = 0
                    for i in range(0, a):
                        for j in range(0, b):
                            if temp[i, j] == 255:
                                count = count + 1
                                countall = countall + 1
                            else:
                                countall = countall + 1
                    ent = count / countall
                    if ent > 0.95:
                        pass
                    else:
                        legend = origin[approx[0][0][1]:approx[1][0][1], approx[0][0][0]:approx[2][0][0]]
                        cv2.drawContours(mask, [approx], -1, (255, 255, 255), -1)
                        cv2.drawContours(mask, [approx], -1, (255, 255, 255), 7)
    return legend, mask


# detect legend's text
def legend_text_detect(image):
    import Gui_define
    #  Pre processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thr = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    kernel = np.uint8([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]])
    opening = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)
    [a, _] = Gui_define.cal_each_x_accumulation(opening)  # 橫向
    [c, _] = Gui_define.cal_each_y_accumulation(thr)  # 直向
    [k3, k4, width, avg] = Gui_define.find_total_bound(a)  # 橫向
    [k1, k2, _, _] = Gui_define.find_total_bound(c)  # 直向
    '''
    # 顯示上下分段
    for i in range(0,len(k1)-1):
        cv2.imshow("1", image[k1[i]:k2[i], :])
        cv2.waitKey()
    cv2.imshow("2",opening)
    cv2.waitKey()
    cv2.imshow("",image[:,6:44])
    cv2.waitKey()
    '''
    # 橫向處理
    k = width.pop(0)
    k_avg = avg.pop(0)
    width[:] = [abs(x - k) for x in width]
    avg[:] = [abs(x - k_avg) for x in avg]
    count = 1
    extra_legend_locate = [0]
    # extra_legend_locate.append(0)
    for i, j in enumerate(width):
        if j < 5:
            if avg[i] < 1:
                count = count + 1
                extra_legend_locate.append(i + 1)
        else:
            pass
    new_range_1 = []
    new_range_2 = []
    legend_num = 0
    if count == 1:
        print("預測無同列的圖例")

        # new_range_1 = []
        # new_range_2 = []
        for i in range(0, len(k1)):
            if i == 0:
                temp = round((k1[i + 1] + k2[i]) / 2)
                new_range_1.append(round(k1[i] / 2))
                new_range_2.append(temp)
            elif i + 1 == len(k1):
                new_range_1.append(temp)
                new_range_2.append(len(c))
            else:
                new_range_1.append(temp)
                temp = round((k2[i] + k1[i + 1]) / 2)
                new_range_2.append(temp)

        for i in range(0, len(k1)):
            text = pytesseract.image_to_string(thr[new_range_1[i]:new_range_2[i], k3[1]:k4[len(k4) - 1]], lang='engB',
                                               config='--psm 6 --oem 1')
            if text != None:
                print(text)
                legend_num = legend_num + 1
        print("共有", legend_num, "種圖例")
    else:
        print("預測有同列的圖例,一列應有", count, "個圖例")
        for i in range(0, len(k1)):
            if i == 0:
                temp = round((k1[i + 1] + k2[i]) / 2)
                new_range_1.append(round(k1[i] / 2))
                new_range_2.append(temp)
            elif i + 1 == len(k1):
                new_range_1.append(temp)
                new_range_2.append(len(c))
            else:
                new_range_1.append(temp)
                temp = round((k2[i] + k1[i + 1]) / 2)
                new_range_2.append(temp)
        for i in range(0, len(k1)):
            for j in range(0, count):
                if j + 1 >= len(extra_legend_locate):
                    text = pytesseract.image_to_string(
                        thr[new_range_1[i]:new_range_2[i], k4[extra_legend_locate[j]]:k4[len(k4) - 1]],
                        lang='engB',
                        config='--psm 6 --oem 1')
                    if text != None:
                        print(text)
                        legend_num = legend_num + 1
                else:
                    text = pytesseract.image_to_string(
                        thr[new_range_1[i]:new_range_2[i], k4[extra_legend_locate[j]]:k3[extra_legend_locate[j + 1]]],
                        lang='engB',
                        config='--psm 6 --oem 1')
                    if text != None:
                        print(text)
                        legend_num = legend_num + 1
        print("共有", legend_num, "種圖例")


def grid_space_detect(image):
    import Gui_define
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    #cv2.imshow("Preprocessing",image)
    #cv2.waitKey()
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
   # plt.plot(a, b)
   # plt.gca().invert_yaxis()
    #plt.show()
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
    c.insert(0,0)
    c.insert(len(c),0)
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

    return x_grid_space, y_grid_space, aa, cc,\
           peaks, peaks1


def remove_x_expected_grid(image, peak, acc_list,
                           space):
    k = []
    row, col, _ =np.shape(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #hsv=cv2.medianBlur(hsv,3)
    thr = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _, thr = cv2.threshold(thr, 200, 255, cv2.THRESH_BINARY)
    for i in peak:
        if abs(round(i / space) - (i / space)) < 0.2:  # 如果間距符合
            if round(i / space) == 0:
                for j in range(0, 4):
                    if acc_list[j] != 0:
                        k.append(j)
            elif round(i / space) == round(row / space):
                for j in range(0, 4):
                    if acc_list[row - 1 - j] != 0:
                        k.append(row - 1 - j)
            else:
                for j in range(0, 4):
                    if acc_list[i - 2 + j] != 0:
                        k.append(i - 2 + j)
    # 補齊有遺漏間隔的網格
    for i in range(0, np.int(round(row / space) + 1)):
        temp = np.int64(i)
        temp = temp * space
        if min(abs(k - temp)) >= 4:
            k.append(temp)
            k.append(temp - 1)
            k.append(temp - 2)
            k.append(temp-3)
            k.append(temp + 1)
            k.append(temp + 2)
            k.append(temp+3)
    k = np.sort(k)
    h_count=[]
    s_count=[]
    v_count=[]
    for i in k:
        for j in range(0, col):
            try:
                if thr[i,j] == 0:
                    v_count.append(hsv[i, j, 2])
            except:
                break

    for i in k:
        for j in range(0, col):

            try:
                if 0<=hsv[i, j, 2] <=  255:
                    image[i, j, :] = 255
            except:
                break


def remove_y_expected_grid(image, peak, acc_list,
                           space):
    k = []
    row, col, _ =np.shape(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for i in peak:
        if abs(round(i / space) - (i / space)) < 0.2:  # 如果間距符合
            if round(i / space) == 0:
                for j in range(0, 4):
                    if acc_list[j] != 0:
                        k.append(j)
            elif round(i / space) == round(col / space):
                for j in range(0, 4):
                    if acc_list[col - 1 - j] != 0:
                        k.append(col - 1 - j)
            else:
                for j in range(0, 4):
                    if acc_list[i - 2 + j] != 0:
                        k.append(i - 2 + j)
    # 補齊有遺漏間隔的網格
    for i in range(0, np.int(round(col / space) + 1)):
        temp = np.int64(i)
        temp = temp * space
        if min(abs(k - temp)) >= 4:
            k.append(temp)
            k.append(temp - 1)
            k.append(temp - 2)
            k.append(temp-3)
            k.append(temp + 1)
            k.append(temp + 2)
            k.append(temp+3)
    k = np.sort(k)
    for i in k:
        for j in range(0, row):
            try:
                if 0<=hsv[j, i, 2] <= 255:
                    image[j, i, :] = 255
            except:
                break


def correct_data(cluster):
    for i in range(0, len(cluster)):
        if i == len(cluster)-1:
                break
        else:
            pre = cluster[i]
            if type(pre) != str:
                if type(cluster[i+1]) == str:
                    for j in range(i+1, len(cluster)):
                        end = cluster[j]
                        if type(end) != str:
                            # print(i, pre)
                            # print(j, end)
                            d = (end - pre)/(j - i)
                            for k in range(i+1, j):
                                pre = pre + d
                                cluster[k] = pre
                            break


def colordist(rgb_1, rgb_2):
    b_1, g_1, r_1 = rgb_1
    b_2, g_2, r_2 = rgb_2
    r_1 = float(r_1)
    g_1 = float(g_1)
    b_1 = float(b_1)
    r_2 = float(r_2)
    g_2 = float(g_2)
    b_2 = float(b_2)
    avg_1 = (b_1 + g_1 + r_1) / 3
    avg_2 = (b_2 + g_2 + r_2) / 3
    gray_1 = np.sqrt((b_1 - avg_1) ** 2 + (g_1 - avg_1) ** 2 + (r_1 - avg_1) ** 2)
    gray_2 = np.sqrt((b_2 - avg_2) ** 2 + (g_2 - avg_2) ** 2 + (r_2 - avg_2) ** 2)
    if abs(gray_1 - gray_2) < 3:
        return 0
    rmean = (r_1 + r_2) / 2
    r = r_1 - r_2
    g = g_1 - g_2
    bl = b_1 - b_2
    return np.sqrt((2 + rmean / 256) * (r ** 2) + 4 * (g ** 2) + (2 + (255 - rmean) / 256) * (bl ** 2))




'''
# 選擇一點
def cv_select_origin(event, x, y, flags, param):
    global coordinate
    if event == cv2.EVENT_LBUTTONDOWN:
        coordinate=(x,y)
        print(image_data[y,x])
        print(x,y)
        imgok=image_data[y - 1:y + 1, x - 1:x + 1, :]
        imgok = cv2.resize(imgok, (300, 300))
        cv2.line(imgok, (140, 150), (160, 150), (0, 0, 255), 1)
        cv2.line(imgok, (150, 140), (150, 160), (0, 0, 255), 1)
        cv2.imshow("123", imgok)

    if event ==cv2.EVENT_MOUSEMOVE:
        try:
            imgk = image_data[y - 5:y + 5, x - 5:x + 5, :]
            imgk = cv2.resize(imgk, (300, 300))
            cv2.line(imgk, (140, 150), (160, 150), (0, 0, 255), 1)
            cv2.line(imgk, (150, 140), (150, 160), (0, 0, 255), 1)
            cv2.imshow("", imgk)
        except:
            pass
'''
'''
def select_origin():
    global coordinate
    coordinate = 1
    cv2.namedWindow('image')
    while coordinate == 1:
        print("選取後按任意鍵確定")
        cv2.setMouseCallback('image', cv_select_origin)
        cv2.imshow('image', image_data)
        cv2.waitKey()
        if coordinate != 1:
            cv2.destroyAllWindows()
    print("您選取的座標為", coordinate)
'''
