import tkinter
import cv2
import PIL.Image, PIL.ImageTk
from easygui import fileopenbox
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema
import Gui_define
import os


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
    if test:
        origin = cv2.imread(fileopenbox())
        cv2.namedWindow("Origin Picture")
        cv2.imshow("Origin Picture", origin)
        test = False
        open_close_text.set("Close Image")

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
        data_region_locate.config(state="active")
        data_region_show.config(state="disabled")
        legend_detect.config(state="disabled")
        legend_show.config(state="disabled")
        legend_removed_show.config(state="disabled")
        grid_detect.config(state="disabled")
        label_detect.config(state="disabled")

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
        test1=True
    #    open_close_text.set("Select and Show Image")


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
            if abs(list[i] - max(list)) < 10:
                target = i
                break
        # Y軸
        image_data = origin[:, target:b]
        # 由右至左找尋有無右邊界
        for i in range(b - 1, 0, -1):
            if abs(list[i] - max(list)) < 10:
                target1 = i
                break
        # Check
        if target1 > 0.8 * b:
            image_data = image_data[:, 0:target1 - target]
        leftbound = target
        rightbound = target1
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
            if abs(list[i - 1] - max(list)) < 10:
                target = i
                break
        # X軸
        image_data = image_data[0:target, :]
        # 由上至下找尋有無上邊界
        for i in range(0, a):
            if abs(list[i] - max(list)) < 10:
                target1 = i
                break
        # Check
        if target1 < 0.9 * a:
            image_data = image_data[target1:target, :]
        else:
            print("找不到上邊界")
        downbound = target
        upbound = target1
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

    cv2.imshow("Legend",legend)
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
        test3=False
        legend_show.config(text="Open Legend")
    else:
        cv2.imshow("Legend", legend)
        test3=True
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
    a, b, c, d, p1, p2 = Gui_define.grid_space_detect(legend_removed)
    grid_removed = legend_removed.copy()
    if a != None:
        Gui_define.remove_x_expected_grid(grid_removed, p1, c, a)
        print("已完成X方向網格刪除")
    if b != None:
        Gui_define.remove_y_expected_grid(grid_removed, p2, d, b)
        print("已完成Y方向網格刪除")
    grid_removed_show_close.config(state="active")
    checkgrid.config(bg='green')
    checkgrid_label.set('Define Grid')
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
    checkvalue_label.set("Define Value")
    checkvalue.config(bg='green')
    print(upbound)


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

# Check_Label
var.set('Not Define Data Region')
checklabel = tkinter.Label(main, textvariable=var, bg='red', padx=10, pady=10, width=20)
checklegend_label.set('Not Define Legend')
checklegend = tkinter.Label(main, textvariable=checklegend_label, bg='red', padx=10, pady=10, width=20)
checkgrid_label.set('Not Define Grid')
checkgrid = tkinter.Label(main, textvariable=checkgrid_label, bg='red', padx=10, pady=10, width=20)
checkvalue_label.set('Not Define Label Value')
checkvalue = tkinter.Label(main, textvariable=checkvalue_label, bg='red', padx=10, pady=10, width=20)
# Button
reopen = tkinter.Button(main, text="Show Origin Image",
                        state="disabled",command=closeimg, width=20, height=1, padx=10, pady=20)
open_close = tkinter.Button(main,textvariable=open_close_text,
                            state="active",command=openfile, width=20, height=1, padx=10, pady=20)
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
label_detect = tkinter.Button(main, text="Label_Detect", command=label_define_fun, state="active", width=20,
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
# hsv_count.grid(row=2,column=0)
# hsv_Individual_display.grid(row=2,column=1)
# origin_select.grid(row=3,column=0)
esc.grid(row=0, column=4, sticky='w')
checklabel.grid(row=5, column=4, sticky='w')
checklegend.grid(row=6, column=4, sticky='w')
checkgrid.grid(row=7, column=4, sticky='w')
checkvalue.grid(row=8, column=4, sticky='w')
main.mainloop()
