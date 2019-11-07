import cv2
import numpy as np
import Gui_define
from matplotlib import pyplot as plt
import xlwt
from tempfile import TemporaryFile
img = cv2.imread("Grid_removed (2).jpg")

[a, b, c] = np.shape(img)  # a=484 b=996,c=3
kernel = np.ones((5, 5), np.uint8)
blur = cv2.blur(img, (3,3))
opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)
cv2.imwrite("blurop.jpg", opening)
lab_img = cv2.cvtColor(opening, cv2.COLOR_BGR2LAB)
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
for i in range(0, cluster_num):
    total_cluster.append([])
    pre_locate.append([""])
    pre_color.append([])
# 將每一行記錄點再次分群，目的是將每一條線條歸類為獨立分群，是為分群對應值，並且對應圖表上的原始資料，可作圖
thr_value = 110
thr1_value = 50
thr_place = 41
thr1_place = 281
y_place = 0
clustered = False
cluster_count = 1


def place_to_value(place):
    if place == thr_place:
        value = thr_value
    else:
        value = round(thr_value - (abs((thr_value - thr1_value)) /
                                   abs((thr1_place - thr_place)) * (j - thr_place)))
    return value


def colordist(rgb_1, rgb_2):
    r_1, g_1, b_1 = rgb_1
    r_2, g_2, b_2 = rgb_2
    r_1 = float(r_1)
    g_1 = float(g_1)
    b_1 = float(b_1)
    r_2 = float(r_2)
    g_2 = float(g_2)
    b_2 = float(b_2)
    rmean = (r_1 + r_2) / 2
    r = r_1 - r_2
    g = g_1 - g_2
    bl = b_1 - b_2
    return np.sqrt((2 + rmean / 256) * (r ** 2) + 4 * (g ** 2) + (2 + (255 - rmean) / 256) * (bl ** 2))


for i in range(len(total_pos)):
    # 空集合不分群
    if total_pos[i] == [None]:
        for n in range(0, cluster_num):
            total_cluster[n].append("")
    # 第一次進入分群閥值定義
    else:
        check_list = [] # 該行確認是否有已排序的類別
        if cluster_count == cluster_num+1:
            clustered = True
        if len(total_pos[i]) == cluster_num:  # 假如該行紀錄點數量剛好等於總分群數
            for j in total_pos[i]:
                value = place_to_value(j)
                if not clustered:  # 未歸類初始化，則直接定義參考值。
                    pre_color[cluster_count-1] = opening[j, i]
                    pre_locate[cluster_count-1] = j
                    total_cluster[cluster_count-1].append(value)
                    print("已定義類別", cluster_count, "初始位置於", i, "行的", pre_locate[cluster_count-1], "座標")
                    cluster_count = cluster_count+1

                else:  # 已歸類參考點。
                    dist_list = []  # 統計各距離
                    for m in range(0, cluster_num):
                        try:
                            dist = abs(pre_locate[m] - j)
                            dist_list.append(dist)
                        except:
                            dist_list.append([])
                   #  place = dist_list.index(min(a for a in dist_list if isinstance))
                    for a, element in enumerate(dist_list):
                        # 找尋dist小且顏色距離小的值。
                        if element < 10:
                            place = dist_list.index(element)
                            if colordist(pre_color[place], opening[j, i]) < 50:
                                pre_color[place] = opening[j, i]
                                total_cluster[place].append(value)
                                check_list.append(place)
                                pre_locate[place] = j
                            else:
                                # 有未分類的值
                                pass
                        else:
                            pass
                    '''
                        place = dist_list.index(min(a for a in dist_list if isinstance))
                        if colordist(pre_color[place], opening[j, i]) < 50:
                            pre_color[place] = opening[j, i]
                            total_cluster[place].append(value)
                            check_list.append(place)
                            pre_locate[place] = j
                            

                    else:
                        print("Error")
'''
        elif len(total_pos[i]) < cluster_num:  # 假如該行紀錄點數量小於總分群數
            for j in total_pos[i]:
                value = place_to_value(j)
                if not clustered:
                    pass
                else:
                    pass
        else:  # 假如該行紀錄點大於總分群數
            for j in total_pos[i]:
                value = place_to_value(j)
                if not clustered:
                    pass
                else:
                    pass


def colordist(rgb_1, rgb_2):
    r_1, g_1, b_1 = rgb_1
    r_2, g_2, b_2 = rgb_2
    r_1 = float(r_1)
    g_1 = float(g_1)
    b_1 = float(b_1)
    r_2 = float(r_2)
    g_2 = float(g_2)
    b_2 = float(b_2)
    rmean = (r_1 + r_2) / 2
    r = r_1 - r_2
    g = g_1 - g_2
    bl = b_1 - b_2
    return np.sqrt((2 + rmean / 256) * (r ** 2) + 4 * (g ** 2) + (2 + (255 - rmean) / 256) * (bl ** 2))


# print(colordist(opening[281, 99], opening[280, 100]))




