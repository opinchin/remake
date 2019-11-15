from PIL import Image
import cv2
import numpy as np
from easygui import fileopenbox
from matplotlib import pyplot as plt
import Gui_define
import pytesseract
import scipy.signal
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes


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
                if 0 <= hsv[j, i, 2] <= 255:
                    image[j, i, :] = 255
            except:
                break

#image=cv2.imread("4_data_regionn.jpg")
image = cv2.imread("Legend Removed_ult.jpg")
#image  = cv2.medianBlur(image,3)
#cv2.imwrite("blur.jpg",blur)
#_, image = Gui_define.legend_locate(image)
bgr = image.copy()
bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
thr = image.copy()
thr = cv2.cvtColor(thr, cv2.COLOR_BGR2GRAY)
_, thr = cv2.threshold(thr, 200, 255, cv2.THRESH_BINARY)


'''
plt.imshow(bgr,'gray'),plt.title('Origin')
plt.show()
plt.subplot(221),plt.imshow(bgr,'gray'),plt.title('Origin')
plt.subplot(222),plt.imshow(thrr,'gray'),plt.title('Origin_Bin')
plt.show()
'''
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
result = image.copy()
a, b, c, d, p1, p2 = grid_space_detect(image)

# 刪除由grid_space_detect預測之網格
#row, col, _ = np.shape(image) #c[row+a*n] , d[col+b*n]
if a!=None:
    remove_x_expected_grid(result, p1, c, a)
    print("已完成X方向網格刪除")
if b!=None:
    remove_y_expected_grid(result, p2, d, b)
    print("已完成Y方向網格刪除")
# 將符合間隔之峰值統計
'''
if a != None:
    k = []
    for i in p1:
        dd=abs(round(i / a) - (i / a))
        if abs(round(i/a)-(i/a))<0.2: #如果間距符合
            if round(i/a) == 0:
                for j in range(0,4):
                    if c[j]!= 0:
                        k.append(j)
            elif round(i/a) == round(row/a):
                for j in range(0,4):
                    if c[row-1-j] != 0:
                        k.append(row-1-j)
            else:
                for j in range(0,4):
                    if c[i-2+j] != 0:
                        k.append(i-2+j)
    # 補齊有遺漏間隔的網格
    for i in range(0, np.int(round(row/a)+1)):
        temp = np.int64(i)
        temp = temp*a
        if min(abs(k-temp))>=4:
            k.append(temp)
            k.append(temp-1)
            k.append(temp-2)
           # k.append(temp-3)
            k.append(temp+1)
            k.append(temp+2)
          #  k.append(temp+3)
    k=np.sort(k)
    for i in k:
        for j in range(0, col):
            if hsv[i,j,2]<250:
                result[i,j,:]=255

'''
result_1 = result.copy()

result_1 = cv2.cvtColor(result_1,cv2.COLOR_RGB2BGR)
plt.subplot(223),plt.imshow(result_1, 'gray'),plt.title('Grid Removed')
result1 = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
#_, result1 = cv2.threshold(result1, 200, 255, cv2.THRESH_BINARY)
imgg = cv2.hconcat([thr,result1])
cv2.imshow("Thr",result_1)
cv2.waitKey()

plt.subplot(224), plt.imshow(result1, 'gray'), plt.title('Grid Removed_Bin')
#plt.show()
#cv2.imshow("grid_removed",reimg)
#a, b = Gui_define.cal_each_y_accumulation(result)
#plt.plot(a)
#plt.show()
#cv2.imshow("c",image)

#cv2.waitKey()

'''
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
row, col = np.shape(image)
cv2.imshow("", image)
cv2.waitKey()
a, b = Gui_define.cal_each_y_accumulation(image)
c, d = Gui_define.cal_each_x_accumulation(image)
# 檢查橫方向之網格線
temp = row / 2
# 過濾過低的累加值
for i in range(0, row):
    if a[i] < temp:
        a[i] = 0
# 找尋峰值
peaks, _ = scipy.signal.find_peaks(a, height=0)
# 紀錄各峰值間隔
dist = []
for i in range(0, len(peaks) - 1):
    temp = peaks[i + 1] - peaks[i]
    dist.append(temp)
# 透過間隔判別網格存在與否
# 統計並分析網格間隔為何
compare = 0
check = False
for i in set(dist):
    temp = []
    # print(i)
    temp[:] = [abs(x - i) for x in dist]
    count = 0
    acc = 0
    for j in temp:
        if j < row / 100:
            count = count + 1
            acc = acc + j
    if count <= 2:
        pass
    else:
        if not check:
            compare = acc
            last_dist = i
            check = True
        elif acc > compare:
            pass
        else:
            compare = acc
            last_dist = i
if check:
    print("預測網格間隔=", last_dist)
else:
    print("預測沒有網格")
# result = Counter(dist)
# temp = result.most_common(1)


new_a = np.array(a)
plt.plot(new_a)
plt.plot(peaks, new_a[peaks], "x")
plt.plot(np.zeros_like(new_a), "--", color="gray")
plt.show()
'''


# legend extract
'''
legend=Gui_define.legend_locate(image)
Gui_define.legend_text_detect(legend)
'''
'''
#image = cv2.imread(fileopenbox())
image = cv2.imread("4.jpg")
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
_, thr = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
kernel = np.ones((3, 3), np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
kernel= np.uint8([[0, 1, 0], [0, 1, 0], [0, 1, 0],[0,1,0]])
erosion = cv2.erode(thr, kernel)
opening = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
[a, _] = Gui_define.cal_each_x_accumulation(opening)  # 橫向
[c, _] = Gui_define.cal_each_y_accumulation(thr)  # 直向
#plt.figure(1)
#plt.plot(c)
#plt.show()
[k3, k4, width, avg] = Gui_define.find_total_bound(a)  # 橫向
[k1, k2, _, _] = Gui_define.find_total_bound(c)  # 直向

'''
# 顯示上下分段
"""
for i in range(0,len(k1)-1):
    cv2.imshow("1", image[k1[i]:k2[i], :])
    cv2.waitKey()
cv2.imshow("2",opening)
cv2.waitKey()
cv2.imshow("",image[:,6:44])
cv2.waitKey()"""
'''
# 橫向處理
k = width.pop(0)
k_avg = avg.pop(0)
width[:] = [abs(x - k) for x in width]
avg[:] = [abs(x - k_avg) for x in avg]
count=1
extra_legend_locate=[]
extra_legend_locate.append(0)
for i, j in enumerate(width):
    if j < 5:
        if avg[i]<1:
            count=count+1
            extra_legend_locate.append(i+1)
    else:
        pass
new_range_1 = []
new_range_2 = []
legend_num = 0
if count ==1:
    print("預測無同列的圖例")

    #new_range_1 = []
    #new_range_2 = []
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
    print("預測有同列的圖例,一列應有",count,"個圖例")
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
        for j in range(0,count):
            if j+1>=len(extra_legend_locate):
                text = pytesseract.image_to_string(
                    thr[new_range_1[i]:new_range_2[i], k4[extra_legend_locate[j]]:k4[len(k4) - 1]],
                    lang='engB',
                    config='--psm 6 --oem 1')
                if text != None:
                    print(text)
                    legend_num = legend_num+1
            else:
                text = pytesseract.image_to_string(
                    thr[new_range_1[i]:new_range_2[i], k4[extra_legend_locate[j]]:k3[extra_legend_locate[j + 1]]],
                    lang='engB',
                    config='--psm 6 --oem 1')
                if text != None:
                    print(text)
                    legend_num = legend_num+1
    print("共有",legend_num  ,"種圖例")
'''
