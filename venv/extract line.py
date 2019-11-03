import cv2
import numpy as np
import Gui_define
from matplotlib import pyplot as plt
import xlwt
from tempfile import TemporaryFile
img = cv2.imread("Grid_removed (2).jpg")
[a, b, c] = np.shape(img)  # a=484 b=996,c=3
kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
hsv_img = cv2.cvtColor(opening, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(opening, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
# 各行的紀錄點位置
total_pos = []
for i in range(0, b):
    pos = []  # 紀錄該行的pixel值
    for j in range(0, a):  # 每一列
        pos.append(thresh[j, i])
    add_none = []  # 欲添加於Total_pos 之 List
    try:
        [k1, k2, k3, k4] = Gui_define.find_total_bound(pos)
        temp = [round((k1[i]+k2[i]-1)/2) for i in range(len(k1))]
        total_pos.append(temp)
    except UnboundLocalError:
        total_pos.append(add_none)
# 若該行紀錄點為空集合，則將其補上None
for i in range(0, np.size(total_pos)):
    if not total_pos[i]:
        total_pos[i] = [None]
# 將每一行位置儲存至Excel分頁一
book = xlwt.Workbook()
sheet1 = book.add_sheet('sheet1')
for i, e in enumerate(total_pos):
    sheet1.write(i, 0, str(total_pos[i]))
name = "new_data.xls"
book.save(name)
book.save(TemporaryFile())

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
hsv_define = []
for i in range(0,cluster_num):
    total_cluster.append([])
    pre_locate.append([""])
    hsv_define.append([])
# 將每一行記錄點再次分群，目的是將每一條線條歸類為獨立分群，是為分群對應值，並且對應圖表上的原始資料，可作圖
thr_value = 110
thr1_value = 50
thr_place = 41
thr1_place = 281
y_place = 0
for i in range(len(total_pos)):
    # 空集合不分群
    if total_pos[i] == [None]:
        for n in range(0, cluster_num):
            total_cluster[n].append("")
    # 進入分群閥值定義
    else:
        if len(total_pos[i]) == cluster_num:
            # 假如該行紀錄點剛好等於總分群數，則直接定義分群HSV閥值
            count = 0
            for j in total_pos[i]:
                if j == thr_place:  # 調整基準對應值
                    value = thr_value
                else:  # 將每一行的資訊做分群
                    value = round(thr_value - (abs((thr_value - thr1_value)) / abs((thr1_place - thr_place)) *
                                               (j - thr_place)))
                hsv_define[count] = (hsv_img[j, i])
                print(count, j)
                count = count+1
            print(hsv_define)
            break




