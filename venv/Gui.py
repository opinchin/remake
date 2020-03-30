import tkinter
import cv2
import PIL.Image, PIL.ImageTk
from easygui import fileopenbox
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema
import Gui_define
import os
import pytesseract
import math
import xlwt
from tempfile import TemporaryFile
from decimal import Decimal

test = True
test1 = True
test2 = True
test3 = True
test4 = True
test5 = True
# global save_path
save_path = 'C:/Users/Burny/PycharmProjects/remake/venv/output'
# 開啟圖片


def openfile():
    global test
    global origin
    global row, col
    if test:
        origin = cv2.imread(fileopenbox())
        cv2.namedWindow("Origin Picture")
        cv2.imshow("Origin Picture", origin)
        test = False
        open_close_text.set("Close Image")
        row, col, _ = np.shape(origin)
        print("Row=", row, "Col=", col)
        # 初始化
#        hsv_count.config(state="disabled")
        checklabel.config(bg='red')
        var.set('Not Define Data Region')
        checklegend.config(bg='red')
        checklegend_label.set('Not Define Legend')
        checkgrid.config(bg='red')
        checkgrid_label.set('Not Define Grid')
        checkvalue.config(bg='red')
        checkvalue_label.set("Not Define Value")
        finish.config(bg='red')
        finish_label.set("Unfinished")
        data_region_locate.config(state="active")
        data_region_show.config(state="disabled")
        legend_detect.config(state="disabled")
        legend_show.config(state="disabled")
        legend_removed_show.config(state="disabled")
        grid_detect.config(state="disabled")
        label_detect.config(state="disabled")
        data_extract.config(state="disabled")


        cv2.imwrite(os.path.join(save_path, 'Origin.jpg'), origin)
    else:
        test = True
        cv2.destroyAllWindows()
        open_close_text.set("Select and Show Image")
        reopen.config(state="active")


# 關閉圖片
def closeimg():
    global test1
    # global origin
    if test1:
        reopen.config(text="Close Origin Image")
        cv2.namedWindow("Origin Picture")
        cv2.imshow("Origin Picture", origin)
        test1=False
    else:
        cv2.destroyWindow("Origin Picture")
        reopen.config(text="Show Origin Image")
        test1 = True


'''
# 統計hsv極值
def high_hsv():
    global high_hsv
    global origin
    global hsv
    global imgk

    imgk = image_data[:, :, [2, 1, 0]]
    hsv = cv2.cvtColor(imgk, cv2.COLOR_RGB2HSV)
    [a, b, c] = np.shape(hsv)
    # 撇除黑灰白的顏色直方圖
    lower_white = np.array([0, 50, 50])
    upper_white = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    hsv_count = []
    for i in range(0, a):
        for j in range(0, b):
            if mask[i, j] != 0:
                value = hsv[i, j, 0]
                hsv_count.append(value)

    z = plt.hist(hsv_count, 256, [0, 256])
    zz = z[0]
    text1 = "Accumulation of the H channel except for Black & White"
    plt.title(text1)

    # 取HSV顏色峰值

    h = len(zz)
    num = []

    for i in range(0, h):
        if zz[i] < 100:
            zz[i] = 0
        num.append(i)

    # plt.plot(num,zz,'k')

    k = argrelextrema(zz, np.greater, order=5)
    [_, k_num] = np.shape(k)

    high_hsv = []
    thr = 5

    # x = np.int(k[0][0])
    # 檢查邊界問題
    for i in range(0, k_num):
        a = int(k[0][i])
        if a + thr > 179:  # a=178
            list = []
            list.append(a)
            for i in range(1, 6):
                if a + i > 179:  # a=178，i=2，須得到0
                    b = a + i - 180
                    list.append(b)
                else:
                    b = a + i
                    list.append(b)
            list1 = zz[a], zz[list[1]], zz[list[2]], zz[list[3]], zz[list[4]], zz[list[5]]
            a = np.where(list1 == max(list1))
            a = np.int(a[0][0])
            a = list[a]
        elif a - thr < 0:  # a=1
            list = []
            list.append(a)
            for i in range(1, 6):
                if a - i < 0:  # a=1 , i=2,須得到179
                    b = 180 + a - i
                    list.append(b)
                else:
                    b = a - i
                    list.append(b)
            list1 = zz[a], zz[list[1]], zz[list[2]], zz[list[3]], zz[list[4]], zz[list[5]]
            a = np.where(list1 == max(list1))
            a = np.int(a[0][0])
            a = list[a]
        high_hsv.append(a)
    # 順序排列H通道峰值
    high_hsv = sorted(set(high_hsv))
    [k_num] = np.shape(high_hsv)
    print("峰值共", k_num, "個")
    hsv_Individual_display.config(state="active")
    plt.show()
'''
'''
# 個別顯示極值
def hsv_show():
    global hsv
    for i in high_hsv:
        x = i
        thr=5
        text1 = 'Origin'
        text2 = 'When H =' + np.str(i) + '+-' + np.str(thr)
        print("閥值=", i)
        if x + thr > 179:  # x=176
            lower = x - thr, 50, 50
            upper = x + thr, 255, 255  # 176+5=181
            carry_upper = (x + thr - 179, 255, 255)
            mask1 = cv2.inRange(hsv, lower, upper)
            mask2 = cv2.inRange(hsv, (0, 50, 50), carry_upper)
            mask = mask1 + mask2
        elif x - thr < 0:  # x=0
            lower = (0, 50, 50)
            upper = (x + thr, 255, 255)
            carry_lower = (180 - (thr - x), 50, 50)
            mask1 = cv2.inRange(hsv, lower, upper)
            mask2 = cv2.inRange(hsv, carry_lower, (180, 255, 255))
            mask = mask1 + mask2
        else:
            lower = (x - thr, 50, 50)
            upper = (x + thr, 255, 255)
            mask = cv2.inRange(hsv, lower, upper)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(imgk, imgk, mask=mask)
        plt.figure(i)
        plt.subplot(1, 2, 1)
        plt.imshow(imgk)
        plt.title(text1)
        plt.subplot(1, 2, 2)
        plt.imshow(res)
        plt.title(text2)
    #    plt.waitforbuttonpress
    plt.show()
'''


# 找尋DataRegion
def dataregion_detect():
    global image_data
    global test2
    global upbound, downbound, leftbound, rightbound
    try:
        gray = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
        ret, gray = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
        [a, b] = np.shape(gray)
        image_data = origin.copy()
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
        # for i in range(0, b):
        #     if abs(list[i] - max(list)) < a / 30:
        #         if abs(list[i + 1] - max(list)) > a / 30:
        #             target = i + 1
        #
        #             break
        leftbound = target
        # Y軸
        image_data = origin[:, target:b]
        # 由右至左找尋有無右邊界
        for i in range(b - 1, 0, -1):
            if abs(list[i] - max(list)) < a/30:
                target1 = i
                break
        # for i in range(b - 1, 0, -1):
        #     if abs(list[i] - max(list)) < a / 30:
        #         if abs(list[i - 1] - max(list)) > a / 30:
        #             target1 = i
        #             break
        # Check
        if target1 > 0.8 * b:
            image_data = image_data[:, 0:target1 - target]
            rightbound = target1
        else:
            rightbound = None
            print("找不到右邊界")

        # rightbound = target1
        # 統計每一列
        list = []
        for i in range(0, a):
            count = 0
            for j in range(0, b):
                if gray[i, j] == 0:
                    count = count + 1
            list.append(count)
        # 由下至上找尋符合邊界(Y軸)
        # for i in range(a, 0, -1):
        #     if abs(list[i - 1] - max(list)) < b/30:
        #         target = i
        #         break
        for i in range(a, 0, -1):
            if abs(list[i - 1] - max(list)) < b / 30:
                if abs(list[i - 2] - max(list)) > b / 30:
                    target = i-1
                    break
        # X軸
        image_data = image_data[0:target, :]
        # 由上至下找尋有無上邊界
        for i in range(0, a):
            # if abs(list[i] - max(list)) < b / 30:
            #     target1 = i
            #     break
            if abs(list[i] - max(list)) < b / 30:
                if abs(list[i + 1] - max(list)) > b / 30:
                    target1 = i+1
                    # target1 = i
                    break
        # Check
        if target1 < 0.9 * a:
            image_data = image_data[target1:target, :]
            upbound = target1
        else:
            upbound = None
            print("找不到上邊界")
        downbound = target

        cv2.imshow("DataRegion",image_data)
        result = tkinter.messagebox.askokcancel("確認資料區域","結果是否於資料區域")
        if result:
            print("正確選取資料區域")
            checklabel.config(bg='green')
            var.set('Defined Data Region')
            legend_detect.config(state="active")
            data_region_show.config(state="active",text="Close Data Region")
            # hsv_count.config(state="active")
            # origin_select.config(state="active")
            test2 = True
            # data region 存檔
            cv2.imwrite(os.path.join(save_path, 'data_region.jpg'),image_data)
        else:
            result1 = tkinter.messagebox.askokcancel("Error", "無法自動偵測是否要自行框選")
            if result1:
                legend_detect.config(state="active")
                r = cv2.selectROI(origin,showCrosshair=False)
                image_data = origin[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
                cv2.imshow("DataRegion",image_data)
                checklabel.config(bg='green')
                var.set('Defined Data Region')
                data_region_show.config(state="active",text="Close Data Region")
                # hsv_count.config(state="active")
                # origin_select.config(state="active")
                test2 = True
            else:
                print("請選擇其他圖片或自行框選正確的資料區域")
    except:
        result1 = tkinter.messagebox.askokcancel("Error", "無法自動偵測是否要自行框選")
        if result1:
            legend_detect.config(state="active")
            r = cv2.selectROI(origin, showCrosshair=False)
            image_data = origin[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
            cv2.imshow("DataRegion", image_data)
            checklabel.config(bg='green')
            var.set('Defined Data Region')
            data_region_show.config(state="active", text="Close Data Region")
            # hsv_count.config(state="active")
            # origin_select.config(state="active")
            test2 = True
        else:
            print("請選擇其他圖片或自行框選正確的資料區域")


# 關閉/開啟 DataRegion
def dataregion_show_close():
    global test2
    if test2:
        cv2.destroyWindow("DataRegion")
        # cv2.destroyWindow("ROI selector")
        test2 = False
        data_region_show.config(text="Open Data Region")
    else:
        cv2.imshow("DataRegion",image_data)
        test2 = True
        data_region_show.config(text="Close Data Region")


# 偵測圖例
def legend_locate():
    global legend
    global legend_removed
    try:
        legend, mask = Gui_define.legend_locate(image_data)
    except:
        print("預測沒有圖例")
        checklegend.config(bg='green')
        checklegend_label.set('Define Legend')
        grid_detect.config(state="active")
        legend_removed = image_data
        return

    cv2.imshow("Legend", legend)
    result = tkinter.messagebox.askokcancel("確認圖例", "結果是否為圖例")
    if result:
        print("圖例正確偵測")
        legend_show.config(state="active")
        Gui_define.legend_text_detect(legend)
        legend_removed = cv2.add(image_data, mask)
        legend_removed_show.config(state="active")
        cv2.imshow("Legend Removed", legend_removed)
        cv2.imwrite(os.path.join(save_path, 'Legend.jpg'), legend)
        cv2.imwrite(os.path.join(save_path, 'Legend Removed.jpg'), legend_removed)
        # cv2.imwrite("Legend Removed.jpg", legend_removed)
    else:
        legend_removed = image_data
        print("沒有圖例")
    grid_detect.config(state="active")
    checklegend.config(bg='green')
    checklegend_label.set('Define Legend')


# 關閉/開啟 Legend
def legend_show_close():
    global test3
    if test3:
        cv2.destroyWindow("Legend")
        test3 = False
        legend_show.config(text="Open Legend")
    else:
        cv2.imshow("Legend", legend)
        test3 = True
        legend_show.config(text="Close Legend")


# 關閉/開啟
def legend_removed_show_close():
    global test4
    if test4:
        cv2.destroyWindow("Legend Removed")
        test4 = False
        legend_removed_show.config(text="Open Legend Removed")
    else:
        cv2.imshow("Legend Removed", legend_removed)
        test4 = True
        legend_removed_show.config(text="Close Legend Removed")


# 偵測網格
def grid_detect_fun():
    global grid_removed
    global grid_x, grid_y
    a, b, c, d, p1, p2 = Gui_define.grid_space_detect(legend_removed)
    grid_removed = legend_removed.copy()
    if a != None:
        grid_x = a
        Gui_define.remove_x_expected_grid(grid_removed, p1, c, a)
        print("已完成X方向網格刪除")
    else:
        grid_x = None
    if b != None:
        grid_y = b
        Gui_define.remove_y_expected_grid(grid_removed, p2, d, b)
        print("已完成Y方向網格刪除")
    else:
        grid_y = None
    grid_removed_show_close.config(state="active")
    checkgrid.config(bg='green')
    checkgrid_label.set('Define Grid')
    label_detect.config(state="active")
    cv2.imshow("Grid_removed", grid_removed)
    cv2.imwrite(os.path.join(save_path, 'Grid_removed.jpg'), grid_removed)


def grid_removed_show_close_fun():
    global test5
    if test5:
        cv2.destroyWindow("Grid_removed")
        test5 = False
        grid_removed_show_close.config(text="Open Grid Removed")
    else:
        cv2.imshow("Grid_removed", grid_removed)
        test5 = True
        grid_removed_show_close.config(text="Close Grid Removed")


def label_define_fun():
    global y_label_place_1, y_label_place_2, y_label_value_1, y_label_value_2
    global x_label_place_1, x_label_place_2, x_label_value_1, x_label_value_2
    x_label = origin[downbound:row, :]
    y_label = origin[:, 0:leftbound]
    x_label_fix = x_label[:, leftbound:rightbound]
    y_label_fix = y_label[upbound:downbound, :]
    if grid_x != None:
        print("網格之案例")
        check_value = []
        check_place = []
        text = pytesseract.image_to_string(y_label_fix, lang='engB', config='--psm 6 --oem 1')
        output = pytesseract.image_to_data(y_label_fix, lang='engB', config='--psm 6 --oem 1',
                                           output_type=pytesseract.Output.DICT)

        num_boxes = len(output['level'])
        for i in range(0, num_boxes):
            try:
                k = float(output['text'][i])
                (x, y, w, h) = (output['left'][i], output['top'][i], output['width'][i],
                                output['height'][i])
                if abs((y + h / 2) / grid_x - round((y + h / 2) / grid_x)) < 0.15:
                    check_value.append(k)
                    check_place.append(int(round((y + h / 2) / grid_x)))
            except ValueError:
                pass
        check = []
        check_place_ = []
        find_or_not = False
        for i in range(0, len(check_value)):
            if find_or_not:
                break
            else:
                check[:] = [abs(x - check_value[i]) for x in check_value]
                k = 0 - check_place[check.index(0)]
                check_place_[:] = [x + k for x in check_place]
                for j in range(0, len(check_value)):
                    try:
                        check[j] = abs(check[j] / check_place_[j])
                    except ZeroDivisionError:
                        pass
                for j in set(check):
                    if check.count(j) >= math.floor(len(check_value) / 2):
                        print("Label Define")
                        thr_value = check_value[check.index(j)]
                        thr_place = grid_x * check_place[check.index(j)]
                        check[check.index(j)] = 0
                        thr1_value = check_value[check.index(j)]
                        thr1_place = grid_x * check_place[check.index(j)]
                        find_or_not = True
                        break
        y_label_place_1 = thr_place
        y_label_value_1 = thr_value
        y_label_place_2 = thr1_place
        y_label_value_2 = thr1_value
        print("Y軸Label參考點一:位於", y_label_place_1, "Value = ", y_label_value_1)
        print("Y軸Label參考點二:位於", y_label_place_2, "Value = ", y_label_value_2)
    else:
        text = pytesseract.image_to_string(y_label_fix, lang='engB', config='--psm 6 --oem 1')
        out1 = pytesseract.image_to_data(y_label_fix, lang='engB', config='--psm 6 --oem 1',
                                         output_type=pytesseract.Output.DICT)
        check_place = []
        check_value = []
        check_place_ = []
        check_dist = []
        num_boxes = len(out1['level'])
        for i in range(0, num_boxes):
            try:
                k = float(out1['text'][i])
                (x, y, w, h) = (out1['left'][i], out1['top'][i], out1['width'][i], out1['height'][i])
                check_value.append(k)
                # check_place.append(y)
                check_place.append(round((y + h / 2)))
                # cv2.rectangle(y_label_fix, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # print(k, "=", x, y, w, h)
            except ValueError:
                pass
        find_or_not = False
        for i in check_place:
            if find_or_not:
                break
            check_dist[:] = [x - i for x in check_place]
            temp = check_dist.index(0)
            if temp + 1 != len(check_place):
                check_dist[:] = [abs(x / check_dist[temp + 1]) for x in check_dist]
                for i in range(0, len(check_dist)):
                    k = abs(check_dist[i] - round(check_dist[i]))
                    if k < 0.1:
                        check_dist[i] = round(check_dist[i])
                    else:
                        check_dist[i] = ''
                k = check_dist.index(0)
                check_value_ = check_value.copy()
                check = check_value.copy()
                check_value_[:] = [abs(Decimal(str(x)) - Decimal(str(check_value[k])))
                                   for x in check_value]
                # print(check_value_)
                # print(check_dist)
                for j in range(0, len(check_value)):
                    try:
                        check[j] = float((check_value_[j]) / check_dist[j])
                    except:
                        pass
                for j in set(check):
                    if check.count(j) >= math.floor(len(check_value) / 2):
                        print("Label Define")
                        thr_value = check_value[check.index(j)]
                        thr_place = check_place[check.index(j)]
                        check[check.index(j)] = 0
                        thr1_value = check_value[check.index(j)]
                        thr1_place = check_place[check.index(j)]
                        find_or_not = True
                        break
                # print(check)
        y_label_place_1 = thr_place
        y_label_value_1 = thr_value
        y_label_place_2 = thr1_place
        y_label_value_2 = thr1_value
        print("Y軸Label參考點一:位於", y_label_place_1, "Value = ", y_label_value_1)
        print("Y軸Label參考點二:位於", y_label_place_2, "Value = ", y_label_value_2)

    if grid_y != None:
        output = pytesseract.image_to_data(x_label_fix, lang='engB', config='--psm 6 --oem 1',
                                           output_type=pytesseract.Output.DICT)
        check_value = []
        check_place = []
        num_boxes = len(output['level'])
        for i in range(0, num_boxes):
            try:
                k = float(output['text'][i])
                (x, y, w, h) = (output['left'][i], output['top'][i], output['width'][i],
                                output['height'][i])
                if abs((x + w / 2) / grid_y - round((x + w / 2) / grid_y)) < 0.15:
                    check_value.append(k)
                    check_place.append(int(round((x + w / 2) / grid_y)))
            except ValueError:
                pass

        check = []
        check_place_ = []
        find_or_not = False
        for i in range(0, len(check_value)):
            if find_or_not:
                break
            else:
             check[:] = [abs(x - check_value[i]) for x in check_value]
             k = 0 - check_place[check.index(0)]
             check_place_[:] = [x + k for x in check_place]
            for j in range(0, len(check_value)):
                try:
                    check[j] = abs(check[j] / check_place_[j])
                except ZeroDivisionError:
                     pass
            for j in set(check):
                if check.count(j) >= math.floor(len(check_value) / 2):
                    print("Label Define")
                    thr_value = check_value[check.index(j)]
                    thr_place = grid_y * check_place[check.index(j)]
                    check[check.index(j)] = 0
                    thr1_value = check_value[check.index(j)]
                    thr1_place = grid_y * check_place[check.index(j)]
                    find_or_not = True
                    break
        x_label_place_1 = thr_place
        x_label_value_1 = thr_value
        x_label_place_2 = thr1_place
        x_label_value_2 = thr1_value
        print("X軸Label參考點一:位於", x_label_place_1, "Value = ", x_label_value_1)
        print("X軸Label參考點二:位於", x_label_place_2, "Value = ", x_label_value_2)
    else:
        out1 = pytesseract.image_to_data(x_label_fix, lang='engB', config='--psm 6 --oem 1',
                                         output_type=pytesseract.Output.DICT)
        check_place = []
        check_value = []
        check_place_ = []
        check_dist = []
        num_boxes = len(out1['level'])
        for i in range(0, num_boxes):
            try:
                k = float(out1['text'][i])
                (x, y, w, h) = (out1['left'][i], out1['top'][i], out1['width'][i], out1['height'][i])
                check_value.append(k)
                check_place.append(round((x + w / 2)))
            except ValueError:
                pass
        find_or_not = False
        for i in check_place:
            if find_or_not:
                break
            check_dist[:] = [x - i for x in check_place]
            temp = check_dist.index(0)
            if temp + 1 != len(check_place):
                check_dist[:] = [abs(x / check_dist[temp + 1]) for x in check_dist]
                for i in range(0, len(check_dist)):
                    k = abs(check_dist[i] - round(check_dist[i]))
                    if k < 0.1:
                        check_dist[i] = round(check_dist[i])
                    else:
                        check_dist[i] = ''
                k = check_dist.index(0)
                check_value_ = check_value.copy()
                check = check_value.copy()
                check_value_[:] = [abs(Decimal(str(x)) - Decimal(str(check_value[k])))
                                   for x in check_value]
                for j in range(0, len(check_value)):
                    try:
                        check[j] = float((check_value_[j]) / check_dist[j])
                    except:
                        pass
                for j in set(check):
                    if check.count(j) >= math.floor(len(check_value) / 2):
                        print("Label Define")
                        thr_value = check_value[check.index(j)]
                        thr_place = check_place[check.index(j)]
                        check[check.index(j)] = 0
                        thr1_value = check_value[check.index(j)]
                        thr1_place = check_place[check.index(j)]
                        find_or_not = True
                        break
                # print(check)
        x_label_place_1 = thr_place
        x_label_value_1 = thr_value
        x_label_place_2 = thr1_place
        x_label_value_2 = thr1_value
        print("X軸Label參考點一:位於", x_label_place_1, "Value = ", x_label_value_1)
        print("X軸Label參考點二:位於", x_label_place_2, "Value = ", x_label_value_2)





    '''
    if grid_x and grid_y != None:
        print("網格之案例")
        check_value = []
        check_place = []
        output = pytesseract.image_to_data(y_label_fix, lang='engB', config='--psm 6 --oem 1',
                                           output_type=pytesseract.Output.DICT)
        num_boxes = len(output['level'])
        for i in range(0, num_boxes):
            try:
                k = float(output['text'][i])
                (x, y, w, h) = (output['left'][i], output['top'][i], output['width'][i],
                                output['height'][i])
                if abs((y + h / 2) / grid_x - round((y + h / 2) / grid_x)) < 0.1:
                    check_value.append(k)
                    check_place.append(int(round((y + h / 2) / grid_x)))
            except ValueError:
                pass
        check = []
        check_place_ = []
        find_or_not = False
        for i in range(0, len(check_value)):
            if find_or_not:
                break
            else:
                check[:] = [abs(x - check_value[i]) for x in check_value]
                k = 0 - check_place[check.index(0)]
                check_place_[:] = [x + k for x in check_place]
                for j in range(0, len(check_value)):
                    try:
                        check[j] = abs(check[j] / check_place_[j])
                    except ZeroDivisionError:
                        pass
                for j in set(check):
                    if check.count(j) >= math.floor(len(check_value) / 2):
                        print("Label Define")
                        thr_value = check_value[check.index(j)]
                        thr_place = grid_x * check_place[check.index(j)]
                        check[check.index(j)] = 0
                        thr1_value = check_value[check.index(j)]
                        thr1_place = grid_x * check_place[check.index(j)]
                        find_or_not = True
                        break
        y_label_place_1 = thr_place
        y_label_value_1 = thr_value
        y_label_place_2 = thr1_place
        y_label_value_2 = thr1_value
        print("Y軸Label參考點一:位於", y_label_place_1, "Value = ", y_label_value_1)
        print("Y軸Label參考點二:位於", y_label_place_2, "Value = ", y_label_value_2)

        output = pytesseract.image_to_data(x_label_fix, lang='engB', config='--psm 6 --oem 1',
                                         output_type=pytesseract.Output.DICT)
        check_value = []
        check_place = []
        num_boxes = len(output['level'])
        for i in range(0, num_boxes):
            try:
                k = float(output['text'][i])
                (x, y, w, h) = (output['left'][i], output['top'][i], output['width'][i],
                                output['height'][i])
                if abs((x + w / 2) / grid_y - round((x + w / 2) / grid_y)) < 0.1:
                    check_value.append(k)
                    check_place.append(int(round((x + w / 2) / grid_y)))
            except ValueError:
                pass
        '''
    '''
            if not output['text'][i].isdigit():
                pass
            else:
                (x, y, w, h) = (output['left'][i], output['top'][i], output['width'][i], output['height'][i])
                if abs((x + w / 2) / grid_y - round((x + w / 2) / grid_y)) < 0.1:
                    check_value.append(int(output['text'][i]))
                    check_place.append(int(round((x + w / 2) / grid_y)))'''
    '''
        check = []
        check_place_ = []
        find_or_not = False
        for i in range(0, len(check_value)):
            if find_or_not:
                break
            else:
                check[:] = [abs(x - check_value[i]) for x in check_value]
                k = 0 - check_place[check.index(0)]
                check_place_[:] = [x + k for x in check_place]
                for j in range(0, len(check_value)):
                    try:
                        check[j] = abs(check[j] / check_place_[j])
                    except ZeroDivisionError:
                        pass
                for j in set(check):
                    if check.count(j) >= math.floor(len(check_value) / 2):
                        print("Label Define")
                        thr_value = check_value[check.index(j)]
                        thr_place = grid_y * check_place[check.index(j)]
                        check[check.index(j)] = 0
                        thr1_value = check_value[check.index(j)]
                        thr1_place = grid_y * check_place[check.index(j)]
                        find_or_not = True
                        break
        x_label_place_1 = thr_place
        x_label_value_1 = thr_value
        x_label_place_2 = thr1_place
        x_label_value_2 = thr1_value
        print("X軸Label參考點一:位於", x_label_place_1, "Value = ", x_label_value_1)
        print("X軸Label參考點二:位於", x_label_place_2, "Value = ", x_label_value_2)
    else:
        text = pytesseract.image_to_string(y_label_fix, lang='engB', config='--psm 6 --oem 1')
        out1 = pytesseract.image_to_data(y_label_fix, lang='engB', config='--psm 6 --oem 1',
                                         output_type=pytesseract.Output.DICT)
        check_place = []
        check_value = []
        check_place_ = []
        check_dist = []
        num_boxes = len(out1['level'])
        for i in range(0, num_boxes):
            try:
                k = float(out1['text'][i])
                (x, y, w, h) = (out1['left'][i], out1['top'][i], out1['width'][i], out1['height'][i])
                check_value.append(k)
                # check_place.append(y)
                check_place.append(round((y + h / 2)))
                cv2.rectangle(y_label_fix, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # print(k, "=", x, y, w, h)
            except ValueError:
                pass
        find_or_not = False
        for i in check_place:
            if find_or_not:
                break
            check_dist[:] = [x - i for x in check_place]
            temp = check_dist.index(0)
            if temp + 1 != len(check_place):
                check_dist[:] = [abs(x / check_dist[temp + 1]) for x in check_dist]
                for i in range(0, len(check_dist)):
                    k = abs(check_dist[i] - round(check_dist[i]))
                    if k < 0.1:
                        check_dist[i] = round(check_dist[i])
                    else:
                        check_dist[i] = ''
                k = check_dist.index(0)
                check_value_ = check_value.copy()
                check = check_value.copy()
                check_value_[:] = [abs(Decimal(str(x)) - Decimal(str(check_value[k])))
                                   for x in check_value]
                # print(check_value_)
                # print(check_dist)
                for j in range(0, len(check_value)):
                    try:
                        check[j] = float((check_value_[j]) / check_dist[j])
                    except:
                        pass
                for j in set(check):
                    if check.count(j) >= math.floor(len(check_value) / 2):
                        print("Label Define")
                        thr_value = check_value[check.index(j)]
                        thr_place = check_place[check.index(j)]
                        check[check.index(j)] = 0
                        thr1_value = check_value[check.index(j)]
                        thr1_place = check_place[check.index(j)]
                        find_or_not = True
                        break
                print(check)
        y_label_place_1 = thr_place
        y_label_value_1 = thr_value
        y_label_place_2 = thr1_place
        y_label_value_2 = thr1_value
        print("Y軸Label參考點一:位於", thr_place, "Value = ", thr_value)
        print("Y軸Label參考點二:位於", thr1_place, "Value = ", thr1_value)

        text = pytesseract.image_to_string(x_label_fix, lang='engB', config='--psm 6 --oem 1')
        out1 = pytesseract.image_to_data(x_label_fix, lang='engB', config='--psm 6 --oem 1',
                                         output_type=pytesseract.Output.DICT)
        check_place = []
        check_value = []
        check_place_ = []
        check_dist = []
        num_boxes = len(out1['level'])
        for i in range(0, num_boxes):
            try:
                k = float(out1['text'][i])
                (x, y, w, h) = (out1['left'][i], out1['top'][i], out1['width'][i], out1['height'][i])
                check_value.append(k)
                check_place.append(round((x + w / 2)))
            except ValueError:
                pass
        find_or_not = False
        for i in check_place:
            if find_or_not:
                break
            check_dist[:] = [x - i for x in check_place]
            temp = check_dist.index(0)
            if temp + 1 != len(check_place):
                check_dist[:] = [abs(x / check_dist[temp + 1]) for x in check_dist]
                for i in range(0, len(check_dist)):
                    k = abs(check_dist[i] - round(check_dist[i]))
                    if k < 0.1:
                        check_dist[i] = round(check_dist[i])
                    else:
                        check_dist[i] = ''
                k = check_dist.index(0)
                check_value_ = check_value.copy()
                check = check_value.copy()
                check_value_[:] = [abs(Decimal(str(x)) - Decimal(str(check_value[k])))
                                   for x in check_value]
                for j in range(0, len(check_value)):
                    try:
                        check[j] = float((check_value_[j]) / check_dist[j])
                    except:
                        pass
                for j in set(check):
                    if check.count(j) >= math.floor(len(check_value) / 2):
                        print("Label Define")
                        thr_value = check_value[check.index(j)]
                        thr_place = check_place[check.index(j)]
                        check[check.index(j)] = 0
                        thr1_value = check_value[check.index(j)]
                        thr1_place = check_place[check.index(j)]
                        find_or_not = True
                        break
                print(check)
        x_label_place_1 = thr_place
        x_label_value_1 = thr_value
        x_label_place_2 = thr1_place
        x_label_value_2 = thr1_value
        print("X軸Label參考點一:位於", thr_place, "Value = ", thr_value)
        print("X軸Label參考點二:位於", thr1_place, "Value = ", thr1_value)
        '''
    data_extract.config(state='active')
    checkvalue_label.set("Define Value")
    checkvalue.config(bg='green')


def data_extract_fun():
    img = grid_removed

    img = cv2.imread("C:/Users/Burny/PycharmProjects/remake/venv/output/Grid_removed.jpg")
    [a, b, c] = np.shape(img)  # a=484 b=996,c=3
    row = a
    col = b
    opening = img
    if grid_x and grid_y != None:

        kernel = np.ones((7, 7), np.uint8)
        blur = cv2.blur(img, (3, 3))
        opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)  # BGR
        opening = cv2.dilate(opening, (3, 3))
    else:
        pass
    gray = cv2.cvtColor(opening, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite("thr.jpg", thresh)
    # 各行的紀錄點位置
    total_pos = []
    for i in range(0, b):
        pos = []  # 紀錄該行的pixel值
        for j in range(0, a):  # 每一列
            pos.append(thresh[j, i])
        add_none = []  # 欲添加於Total_pos 之 List
        [k1, k2, k3, k4] = Gui_define.find_total_bound(pos)
        temp = [round((k1[i] + k2[i] - 1) / 2) for i in range(len(k1))]
        total_pos.append(temp)
        # try:
        #     [k1, k2, k3, k4] = Gui_define.find_total_bound(pos)
        #     temp = [round((k1[i] + k2[i] - 1) / 2) for i in range(len(k1))]
        #     total_pos.append(temp)
        # except UnboundLocalError:
        #     total_pos.append(add_none)
    # 若該行紀錄點為空集合，則將其補上None
    for i in range(0, np.size(total_pos)):
        if not total_pos[i]:
            total_pos[i] = [None]
    # 將每一行位置儲存至Excel分頁一
    book = xlwt.Workbook()
    sheet1 = book.add_sheet('Total_Position')
    for i, e in enumerate(total_pos):
        sheet1.write(i, 0, str(total_pos[i]))
    name = "new_data.xls"
    try:
        book.save(name)
        book.save(TemporaryFile())
    except PermissionError:
        print("請關閉Excel後存檔")
        pass
    cluster_num = 0
    pre_cluster_num = 0
    # 取出所有記錄點結果之數量的最大值
    for i in range(len(total_pos)):
        a = len(total_pos[i])
        if a > cluster_num:
            count = 0
            for j in range(len(total_pos)):
                if a == len(total_pos[j]):
                    count = count + 1
            if count < len(total_pos) / 10:
                cluster_num = pre_cluster_num
            else:
                pre_cluster_num = a
                cluster_num = a
    print("分成", cluster_num, "類")

    # 分群對應值 與 分群參考點 之初始化
    total_cluster = []
    pre_locate = []
    pre_color = []
    temp_locate = []
    x_value = []
    total_cluster_pos = []
    pre_locate_col = []
    for i in range(0, cluster_num):
        total_cluster.append([])
        pre_locate.append([""])
        temp_locate.append([""])
        pre_color.append([])
        total_cluster_pos.append([])
        pre_locate_col.append([""])
    clustered = False
    cluster_count = 1

    def place_to_value(loc):
        if loc == y_label_place_1:
            value_ = y_label_value_1
        else:
            value_ = y_label_value_1 - (abs((y_label_value_1 - y_label_value_2)) /
                                       abs((y_label_place_1 - y_label_place_2)) * (loc - y_label_place_1))
        return np.float(value_)

    def x_label_to_value(loc):

        if loc == x_label_place_1:
            value_ = x_label_value_1
        else:
            value_ = x_label_value_1 - (abs((x_label_value_1 - x_label_value_2)) /
                                       abs((x_label_place_1 - x_label_place_2)) * (x_label_place_1 - loc))
        return np.float(value_)

    # def colordist(rgb_1, rgb_2):
    #     b_1, g_1, r_1 = rgb_1
    #     b_2, g_2, r_2 = rgb_2
    #     r_1 = float(r_1)
    #     g_1 = float(g_1)
    #     b_1 = float(b_1)
    #     r_2 = float(r_2)
    #     g_2 = float(g_2)
    #     b_2 = float(b_2)
    #     if abs(b_1-g_1) and abs(g_1 - r_1) and (b_1 - r_1) < 20:
    #         avg_1 = (b_1 + g_1 + r_1) / 3
    #         avg_2 = (b_2 + g_2 + r_2) / 3
    #         gray_1 = np.sqrt((b_1 - avg_1) ** 2 + (g_1 - avg_1) ** 2 + (r_1 - avg_1) ** 2)
    #         gray_2 = np.sqrt((b_2 - avg_2) ** 2 + (g_2 - avg_2) ** 2 + (r_2 - avg_2) ** 2)
    #         if abs(gray_1 - gray_2) < 3:
    #             return 0
    #     rmean = (r_1 + r_2) / 2
    #     r = r_1 - r_2
    #     g = g_1 - g_2
    #     bl = b_1 - b_2
    #     return np.sqrt((2 + rmean / 256) * (r ** 2) + 4 * (g ** 2) + (2 + (255 - rmean) / 256) * (bl ** 2))
    #
    # def labdist(lab_1, lab_2):
    #     l_1, a_1, b_1 = lab_1
    #     l_2, a_2, b_2 = lab_2
    #     return np.sqrt((l_1 - l_2) ** 2 + (a_1 - a_2) ** 2 + (b_1 - b_2) ** 2)
    #
    # def expect_locate(in_which_cluster, in_which_col):
    #     ones = False
    #
    #     # for i in range(in_which_col, len(total_pos)):
    #     #     if len(total_pos[i]) == cluster_num:
    #     #         if ones:
    #     #             ones = False
    #     #             break
    #     #         for j in total_pos[i]:
    #     #             color_dist = colordist(pre_color[in_which_cluster], opening[j, i])
    #     #             if color_dist < 125 and not ones:
    #     #                 end = j
    #     #                 end_place = i - 1  # 38
    #     #                 ones = True
    #     #                 count = i - (in_which_col - 1)
    #     #                 break
    #
    #     if in_which_col == len(total_pos):
    #         return
    #     for i in range(in_which_col, len(total_pos)):
    #         dist = []
    #         if i == len(total_pos)-1:
    #             return
    #         if len(total_pos[i]) == cluster_num:
    #             for j in total_pos[i]:
    #                 color_dist = colordist(pre_color[in_which_cluster], opening[j, i])
    #                 dist.append(color_dist)
    #             place = dist.index(min(dist))
    #             end = total_pos[i][place]
    #             end_place = i - 1
    #             count = i - (in_which_col - 1)
    #             break
    #
    #     for i in range(0, count):
    #         dist = []
    #         dist[:] = [abs(end - x) for x in total_pos[end_place]]
    #         place = dist.index(min(dist))  # 1
    #         end = total_pos[end_place][place]
    #         end_place = end_place - 1
    #     return end

    # 將每一行記錄點再次分群，目的是將每一條線條歸類為獨立分群，是為分群對應值，並且對應圖表上的原始資料，可作圖
    for i in range(len(total_pos)):
        # 空集合不分群
        if cluster_count == cluster_num + 1:
            break
        value = x_label_to_value(i)
        x_value.append(value)
        # try:
        #     value = x_label_to_value(i)
        #     x_value.append(value)
        # except:
        #     pass
        if total_pos[i] == [None]:
            for n in range(0, cluster_num):
                total_cluster[n].append("")
                total_cluster_pos[n].append("")
        # 第一次進入分群閥值定義
        else:
            if len(total_pos[i]) == cluster_num:  # 假如該行紀錄點數量剛好等於總分群數
                for j in total_pos[i]:
                    value = place_to_value(j)
                    if not clustered:  # 未歸類初始化，則直接定義參考值。
                        if len(total_pos[i + 1]) == cluster_num and len(total_pos[i + 2]) == cluster_num:
                            pre_color[cluster_count - 1] = opening[j, i]
                            pre_locate[cluster_count - 1] = j
                            pre_locate_col[cluster_count - 1] = i
                            total_cluster[cluster_count - 1].append(value)
                            total_cluster_pos[cluster_count - 1].append(j)
                            print("已定義類別", cluster_count, "初始位置於", i, "行的", pre_locate[cluster_count - 1], "座標")
                            cluster_count = cluster_count + 1
                            locate = i
                    else:  # 已歸類參考點。
                        pass
            elif len(total_pos[i]) < cluster_num:  # 假如該行紀錄點數量小於總分群數
                for n in range(0, cluster_num):
                    total_cluster[n].append("")
                    total_cluster_pos[n].append("")
                    continue
                for j in total_pos[i]:
                    value = place_to_value(j)
                    if not clustered:
                        pass
                    else:
                        pass
            else:  # 假如該行紀錄點大於總分群數
                for n in range(0, cluster_num):
                    total_cluster[n].append("")
                    total_cluster_pos[n].append("")
                for j in total_pos[i]:
                    value = place_to_value(j)
                    if not clustered:
                        pass
                    else:
                        pass
    check_cross = False
    count_none = 0
    cross_num = 0
    # 根據Total_pos開始分類
    for i in range(locate + 1, len(total_pos)):
        value = x_label_to_value(i)
        x_value.append(value)
        # if 太多None再後半段 則視為數據中止
        if total_pos[i] == [None] and i > 4 / 5 * len(total_pos):
            count_none = count_none + 1
        if total_pos[i] == [None]:
            for n in range(0, cluster_num):
                total_cluster[n].append("")
                total_cluster_pos[n].append("")
        else:
            check_list = []
            check_count = 0

            # check_if_close
            check_close_list = []
            check_close = False
            for j in range(0, cluster_num):
                check_close_list[:] = [abs(x - pre_locate[j]) for x in pre_locate]
                check_close_num = 0
                for k in check_close_list:
                    if k < row / 15:
                        check_close_num = check_close_num + 1
                        if k == 0:
                            pass
                        else:
                            close_place = check_close_list.index(k)
                if check_close_num == 2:
                    check_close = True
                    close_place_1 = j
                    # print("Close in ", i, close_place_1, close_place)
                    break

            # check cross
            check_cross_num = 0
            error_cross = False
            if cross_num != 0 and len(total_pos[i]) >= cluster_num:
                cross_num = 0

            if len(total_pos[i]) < cluster_num and check_close:

                pos = []
                for j in range(0, a):  # 每一列
                    pos.append(thresh[j, i])
                [k1, k2, k3, k4] = Gui_define.find_total_bound(pos)

                if check_cross:
                    pass
                else:
                    if cross_num == 0:
                        pre_1 = pre_locate[close_place_1]  # 376
                        pre = pre_locate[close_place]  # 363

                        # 2/15
                        pos = []
                        for j in range(0, row):  # 前一列
                            pos.append(thresh[j, i - 1])
                        [k1, k2, k3, k4] = Gui_define.find_total_bound(pos)

                        pos = []
                        for j in range(0, row):  # 當前列
                            pos.append(thresh[j, i])
                        [k1_, k2_, k3_, k4] = Gui_define.find_total_bound(pos)
                        temp1 = None
                        for j in total_pos[i]:
                            if pre > pre_1:
                                if j in range(pre_1, pre + 1):
                                    place = total_pos[i].index(j)
                                    temp1 = j
                                    break
                            else:
                                if j in range(pre, pre_1 + 1):
                                    place = total_pos[i].index(j)
                                    temp1 = j
                                    break
                        if temp1 == None:
                            pass
                        else:
                            for j in total_pos[i - 1]:
                                if abs(temp1 - j) < 5:
                                    place_ = total_pos[i - 1].index(j)
                                    # print("誤判為Cross in", i)
                                    if abs(k3[place_] - k3_[place]) <= math.ceil(0.1 * k3_[place]):
                                        error_cross = True
                                        print("誤判為Cross in", i)
                                        break

                    # double check if error  or not
                    double_check = []
                    double_check_count = 0
                    for j in total_pos[i]:
                        dist_list = []
                        for k in range(0, cluster_num):
                            dist = abs(pre_locate[k] - j)
                            dist_list.append(dist)
                        if min(dist_list) > 20:
                            pass
                        else:
                            place = dist_list.index(min(dist_list))
                            double_check.append(place)
                    if close_place in double_check:
                        double_check_count = double_check_count + 1
                    if close_place_1 in double_check:
                        double_check_count = double_check_count + 1
                    if double_check_count == 2:
                        cross_num = 0
                    else:
                        if pre_1 > pre:
                            for l in total_pos[i]:
                                if l in range(pre, pre_1):
                                    check_cross_num = check_cross_num + 1
                        else:
                            for l in total_pos[i]:
                                if l in range(pre_1, pre):
                                    check_cross_num = check_cross_num + 1
            if check_cross_num == 1:
                cross_num = cross_num + 1
            if error_cross:
                cross_num = 0
                error_cross = False
            if cross_num >= 2:
                if pre > pre_1:
                    cross_slope = -1
                else:
                    cross_slope = 1
                if abs(pre_locate_col[close_place_1] - pre_locate_col[close_place]) > col * 0.1:
                    pass
                else:
                    check_cross = True
                    cross_num = 0
                    print("Cross in ", i, close_place, close_place_1)

            # 考量前者Cross狀況進行分類
            for j in total_pos[i]:
                if count_none > 10:
                    break
                if len(total_pos[i]) <= cluster_num:
                    #  特殊案例 確認交叉後 第一次歸類
                    if check_cross and len(total_pos[i]) == cluster_num:
                        temp = []
                        tempp = []
                        temp[:] = [j - pre_locate[close_place] for j in total_pos[i]]
                        tempp[:] = [abs(j - pre_locate[close_place]) for j in total_pos[i]]  # 212 65 51
                        abs_temp = tempp.copy()
                        place = tempp.index(min(tempp))
                        pos = total_pos[i][place]
                        tempp.pop(place)
                        place1 = abs_temp.index(min(tempp))
                        pos1 = total_pos[i][place1]
                        # Error correct 如果 temp = [-3, 3, 211]，abs後則會有兩相同的值之誤判
                        if abs(pos - pos1) > row * 0.15:
                            break
                        if place == place1:
                            place1 = temp.index(min(tempp))
                            pos1 = total_pos[i][place1]
                        if pos > pos1:
                            pos_slope = 1
                        else:
                            pos_slope = -1
                        value = place_to_value(pos)
                        value1 = place_to_value(pos1)
                        check_list.append(close_place)
                        check_list.append(close_place_1)
                        if cross_slope < 0:  # 遞減
                            if pos_slope < 0:
                                temp_locate[close_place] = total_pos[i][place]
                                temp_locate[close_place_1] = total_pos[i][place1]
                                pre_locate_col[close_place] = i
                                pre_locate_col[close_place_1] = i
                                total_cluster[close_place].append(value)
                                total_cluster[close_place_1].append(value1)
                                total_cluster_pos[close_place].append(total_pos[i][place])
                                total_cluster_pos[close_place_1].append(total_pos[i][place1])
                            else:
                                temp_locate[close_place] = total_pos[i][place1]
                                temp_locate[close_place_1] = total_pos[i][place]
                                pre_locate_col[close_place] = i
                                pre_locate_col[close_place_1] = i
                                total_cluster[close_place].append(value1)
                                total_cluster[close_place_1].append(value)
                                total_cluster_pos[close_place].append(total_pos[i][place1])
                                total_cluster_pos[close_place_1].append(total_pos[i][place])
                        else:
                            if pos_slope > 0:
                                temp_locate[close_place] = total_pos[i][place]
                                temp_locate[close_place_1] = total_pos[i][place1]
                                pre_locate_col[close_place] = i
                                pre_locate_col[close_place_1] = i
                                total_cluster[close_place].append(value)
                                total_cluster[close_place_1].append(value1)
                                total_cluster_pos[close_place].append(total_pos[i][place])
                                total_cluster_pos[close_place_1].append(total_pos[i][place1])
                            else:
                                temp_locate[close_place] = total_pos[i][place1]
                                temp_locate[close_place_1] = total_pos[i][place]
                                pre_locate_col[close_place] = i
                                pre_locate_col[close_place_1] = i
                                total_cluster[close_place].append(value1)
                                total_cluster[close_place_1].append(value)
                                total_cluster_pos[close_place].append(total_pos[i][place1])
                                total_cluster_pos[close_place_1].append(total_pos[i][place])
                        for j in total_pos[i]:
                            dist_list = []
                            value = place_to_value(j)
                            for k in range(0, cluster_num):
                                dist = abs(pre_locate[k] - j)
                                dist_list.append(dist)
                            for a, element in enumerate(dist_list):
                                if dist_list.index(min(dist_list)) == a:
                                    place = a
                                    try:
                                        try_y = check_list.index(place)
                                    except ValueError:
                                        check_list.append(place)
                                        check_count = check_count + 1
                                        temp_locate[place] = j
                                        pre_locate_col[place] = i
                                        total_cluster[place].append(value)
                                        total_cluster_pos[place].append(j)
                        print("fff", i)
                        check_cross = False
                        break
                dist_list = []
                value = place_to_value(j)
                for k in range(0, cluster_num):
                    dist = abs(pre_locate[k] - j)
                    dist_list.append(dist)
                for a, element in enumerate(dist_list):
                    if dist_list.index(min(dist_list)) == a:
                        if min(dist_list) > row / 20:
                            break
                        place = a
                        try:
                            try_y = check_list.index(place)
                            fix = None
                            if fix == None:
                                pass
                            else:
                                value = place_to_value(fix)
                                total_cluster[place].pop()
                                total_cluster[place].append(value)
                                total_cluster_pos[place].pop()
                                total_cluster_pos[place].append(fix)
                                temp_locate[place] = fix
                                pre_locate_col[place] = i
                        except ValueError:
                            if check_cross:
                                if place == close_place or place == close_place_1:
                                    break
                            check_list.append(place)
                            check_count = check_count + 1
                            temp_locate[place] = j
                            pre_locate_col[place] = i
                            total_cluster[place].append(value)
                            total_cluster_pos[place].append(j)
                            break
            for k in range(0, cluster_num):
                if temp_locate[k] == [""]:
                    temp_locate[k] = pre_locate[k]
                pre_locate[k] = temp_locate[k]
            for l in range(0, cluster_num):
                try:
                    check_list.index(l)
                except ValueError:
                    total_cluster[l].append("")
                    total_cluster_pos[l].append("")
            # 整合前程式
            # for j in total_pos[i]:
            #     dist_list = []
            #     color_dist_list = []
            #     value = place_to_value(j)
            #     for k in range(0, cluster_num):
            #         dist = abs(pre_locate[k] - j)
            #         color_dist = colordist(pre_color[k], opening[j, i])
            #         dist_list.append(dist)
            #         color_dist_list.append(color_dist)
            #
            #     if len(total_pos[i]) == cluster_num:
            #         place = color_dist_list.index(min(color_dist_list))
            #         if min(color_dist_list) < 125 and dist_list[place] < 25:
            #             check_list.append(place)
            #             temp_locate[place] = j
            #             total_cluster[place].append(value)
            #             continue
            #     for a, element in enumerate(color_dist_list):
            #         if element < 125 and dist_list[a] < 25:
            #             place = a
            #             try:
            #                 try_y = check_list.index(place)
            #                 try:
            #                     if abs(fix - pre_locate[a]) < 20:
            #                         value = place_to_value(fix)
            #                         total_cluster[place].pop()
            #                         total_cluster[place].append(value)
            #                         temp_locate[place] = fix
            #                 except:
            #                     pass
            #
            #             except ValueError:
            #
            #                 check_list.append(place)
            #                 check_count = check_count + 1
            #                 temp_locate[place] = j
            #                 total_cluster[place].append(value)
            #                 break
            #
            #
            # for l in range(0, cluster_num):
            #     if temp_locate[l] == ['']:
            #         pass
            #     else:
            #         pre_locate[l] = temp_locate[l]
            #     try:
            #         check_list.index(l)
            #     except ValueError:
            #         total_cluster[l].append("")
    # 存檔
    sheet2 = book.add_sheet('after extracting')
    for i, e in enumerate(x_value):
        sheet2.write(i, 0, e)

    for i in range(0, cluster_num):
        Gui_define.correct_data(total_cluster[i])
        # correct_data(total_cluster[i])
    for k in range(0, cluster_num):
        col = k + 1
        for i, e in enumerate(total_cluster[k]):
            if type(e) != str:
                sheet2.write(i, col, e)
            else:
                sheet2.write(i, col, str(total_cluster[k][i]))
    name = "new_data.xls"
    try:
        book.save(name)
        book.save(TemporaryFile())
    except PermissionError:
        print("請關閉Excel後存檔")
        pass

    sheet3 = book.add_sheet('sheet3')
    for k in range(0, cluster_num):
        col = k
        for i, e in enumerate(total_cluster_pos[k]):
            if type(e) != str:
                sheet3.write(i, col, e)
            else:
                sheet3.write(i, col, str(total_cluster_pos[k][i]))
    name = "new_data.xls"
    try:
        book.save(name)
        book.save(TemporaryFile())
    except PermissionError:
        print("請關閉Excel後存檔")
        pass
    # try:
    #     for i, e in enumerate(x_value):
    #         sheet2.write(i, 0, e)
    # except:
    #     print(x_value)
    #     pass
    # for i in range(0, cluster_num):
    #     Gui_define.correct_data(total_cluster[i])
    # for k in range(0, cluster_num):
    #     col = k + 1
    #     for i, e in enumerate(total_cluster[k]):
    #         if type(e) == float:
    #             sheet2.write(i, col, e)
    #         else:
    #             sheet2.write(i, col, str(total_cluster[k][i]))
    # name = "new_data.xls"
    # try:
    #     book.save(name)
    #     book.save(TemporaryFile())
    # except PermissionError:
    #     print("請關閉Excel後存檔")
    #     pass
    print("Finish Extracting")

    finish_label.set("Finished")
    finish.config(bg='green')


# ESC KEY
def escape():
    global main
    main.quit()


# Create a window & initialization
main = tkinter.Tk()
main.resizable(width=False,height=False)
var = tkinter.StringVar()
checklegend_label = tkinter.StringVar()
open_close_text = tkinter.StringVar()
checkgrid_label = tkinter.StringVar()
checkvalue_label = tkinter.StringVar()
finish_label = tkinter.StringVar()

# Check_Label
var.set('Not Define Data Region')
checklabel = tkinter.Label(main, textvariable=var, bg='red', padx=10, pady=10, width=20)
checklegend_label.set('Not Define Legend')
checklegend = tkinter.Label(main, textvariable=checklegend_label, bg='red', padx=10, pady=10, width=20)
checkgrid_label.set('Not Define Grid')
checkgrid = tkinter.Label(main, textvariable=checkgrid_label, bg='red', padx=10, pady=10, width=20)
checkvalue_label.set('Not Define Label Value')
checkvalue = tkinter.Label(main, textvariable=checkvalue_label, bg='red', padx=10, pady=10, width=20)
finish_label.set('Unfinished')
finish = tkinter.Label(main, textvariable=finish_label, bg='red', padx=10, pady=10, width=20)

# Button
reopen = tkinter.Button(main, text="Show Origin Image",
                        state="disabled", command=closeimg, width=20, height=1, padx=10, pady=20)
open_close = tkinter.Button(main,textvariable=open_close_text,
                            state="active", command=openfile, width=20, height=1, padx=10, pady=20)
# hsv_Individual_display=tkinter.Button(main,text="Individual display",command=hsv_show,state="disabled",
# width=20, height=1, padx=10, pady=20)
# hsv_count=tkinter.Button(main,text="hsv",command=tryt(),state="disabled", width=20, height=1, padx=10, pady=20)
open_close_text.set("Select and Show Image")
data_region_locate = tkinter.Button(main, text="Data_region_detect",command=dataregion_detect, width=20,
                                    height=1, padx=10, pady=20, state="disabled")
data_region_show = tkinter.Button(main, text="Close_Data_region", command=dataregion_show_close,
                                  state="disabled", width=20, height=1, padx=10, pady=20)
# origin_select=tkinter.Button(main,text="SelectOrigin",command=select_origin,state="disabled",
# width=20, height=1, padx=10, pady=20)
legend_detect = tkinter.Button(main, text="Legend_Detect", command=legend_locate,
                               state="disabled", width=20, height=1, padx=10, pady=20)
legend_show = tkinter.Button(main, text="Close_Legend", command=legend_show_close,
                             state="disabled", width=20, height=1, padx=10, pady=20)
legend_removed_show = tkinter.Button(main, text="Close_Remove_Legend", command=legend_removed_show_close,
                                     state="disabled", width=20, height=1, padx=10, pady=20)
grid_detect = tkinter.Button(main, text="Grid_Detect", command=grid_detect_fun,
                             state="disabled",width=20, height=1, padx=10, pady=20)
grid_removed_show_close = tkinter.Button(main, text="Close_Grid_Legend", command=grid_removed_show_close_fun,
                                         state="disabled", width=20, height=1, padx=10, pady=20)
label_detect = tkinter.Button(main, text="Label_Detect", command=label_define_fun, state="disabled", width=20,
                              height=1, padx=10, pady=20)
data_extract = tkinter.Button(main, text="Data_Extract", command=data_extract_fun, state="disabled", width=20,
                              height=1, padx=10, pady=20)
esc = tkinter.Button(main, text="Esc", command=escape,
                     state="active", width=20, height=1, padx=10, pady=20)

# Location

open_close.grid(row=0, column=0)
reopen.grid(row=0, column=1)
data_region_locate.grid(row=1, column=0)
data_region_show.grid(row=1, column=1)
legend_detect.grid(row=2, column=0)
legend_show.grid(row=2, column=1)
legend_removed_show.grid(row=2, column=2)
grid_detect.grid(row=3, column=0)
grid_removed_show_close.grid(row=3, column=1)
label_detect.grid(row=4, column=0)
data_extract.grid(row=5, column=0)
# hsv_count.grid(row=2,column=0)
# hsv_Individual_display.grid(row=2,column=1)
# origin_select.grid(row=3,column=0)
esc.grid(row=0, column=4, sticky='w')
checklabel.grid(row=6, column=4, sticky='w')
checklegend.grid(row=7, column=4, sticky='w')
checkgrid.grid(row=8, column=4, sticky='w')
checkvalue.grid(row=9, column=4, sticky='w')
finish.grid(row=10 , column=4, sticky='w')
main.mainloop()
