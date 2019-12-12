import cv2
import numpy as np
import Gui_define
from matplotlib import pyplot as plt
import xlwt
from tempfile import TemporaryFile
import scipy.signal
import pytesseract


def correct_data(cluster):
    for i in range(0, len(cluster)):
        if i == len(cluster)-1:
            break
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


def dataregion_detect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, gray = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
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
        if abs(list[i] - max(list)) < 10:
            target = i
            break
    left_bound = target
    # X軸
    image_data = img[:, target:b]
    # 由右至左找尋有無右邊界
    for i in range(b - 1, 0, -1):
        if abs(list[i] - max(list)) < 10:
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
        if abs(list[i - 1] - max(list)) < 10:
            target = i
            break
    down_bound = target
    # Y軸
    image_data = image_data[0:target, :]
    # 由上至下找尋有無上邊界
    for i in range(0, a):
        if abs(list[i] - max(list)) < 10:
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


img = cv2.imread("Grid_removed.jpg")
[a, b, c] = np.shape(img)  # a=484 b=996,c=3
kernel = np.ones((5, 5), np.uint8)
#blur = cv2.medianBlur(img, 3)
blur = cv2.blur(img, (3, 3))
opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)  # BGR
# opening = cv2.erode(opening,kernel)
opening = cv2.dilate(opening, (5, 5))
lab_img = cv2.cvtColor(opening, cv2.COLOR_BGR2LAB)
hsv_img = cv2.cvtColor(opening, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(opening, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("f", thresh)
cv2.waitKey()
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
    '''
    try:
        [k1, k2, k3, k4] = Gui_define.find_total_bound(pos)
        temp = [round((k1[i]+k2[i]-1)/2) for i in range(len(k1))]
        total_pos.append(temp)
    except UnboundLocalError:
        total_pos.append(add_none)'''

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
x_value = []
for i in range(0, cluster_num):
    # x_value.append([])
    total_cluster.append([])
    pre_locate.append([""])
    pre_color.append([])

thr_value = 0.7
thr1_value = 0.6
thr_place = 164
thr1_place = 246

x_label_value_1 = 20
x_label_place_1 = 271
x_label_value_2 = 25
x_label_place_2 = 406

clustered = False
cluster_count = 1


def place_to_value(place):
    if place == thr_place:
        value = thr_value
    else:
        value = thr_value - (abs((thr_value - thr1_value)) /
                                   abs((thr1_place - thr_place)) * (place - thr_place))
    return value


def x_label_to_value(place):
    if place == x_label_place_1:
        value = x_label_value_1
    else:
        value = x_label_value_1 - (abs((x_label_value_1 - x_label_value_2)) /
                             abs((x_label_place_1 - x_label_place_2)) * (x_label_place_1 - place))
    return value


def colordist(rgb_1, rgb_2):
    b_1, g_1, r_1 = rgb_1
    b_2, g_2, r_2 = rgb_2
    r_1 = float(r_1)
    g_1 = float(g_1)
    b_1 = float(b_1)
    r_2 = float(r_2)
    g_2 = float(g_2)
    b_2 = float(b_2)
    avg_1 = (b_1 + g_1 + r_1)/3
    avg_2 = (b_2 + g_2 + r_2)/3
    gray_1 = np.sqrt((b_1 - avg_1) ** 2 + (g_1 - avg_1) ** 2 + (r_1 - avg_1) ** 2)
    gray_2 = np.sqrt((b_2 - avg_2) ** 2 + (g_2 - avg_2) ** 2 + (r_2 - avg_2) ** 2)
    if abs(gray_1 - gray_2) < 3:
        return 0
    rmean = (r_1 + r_2) / 2
    r = r_1 - r_2
    g = g_1 - g_2
    bl = b_1 - b_2
    return np.sqrt((2 + rmean / 256) * (r ** 2) + 4 * (g ** 2) + (2 + (255 - rmean) / 256) * (bl ** 2))


def labdist(lab_1, lab_2):
    l_1, a_1, b_1 = lab_1
    l_2, a_2, b_2 = lab_2
    return np.sqrt((l_1-l_2)**2 + (a_1-a_2)**2 + (b_1-b_2)**2)


# 將每一行記錄點再次分群，目的是將每一條線條歸類為獨立分群，是為分群對應值，並且對應圖表上的原始資料，可作圖
for i in range(len(total_pos)):
    # 空集合不分群
    if cluster_count == cluster_num + 1:
        break
    value = x_label_to_value(i)
    x_value.append(value)
    if total_pos[i] == [None]:
        for n in range(0, cluster_num):
            total_cluster[n].append("")
    # 第一次進入分群閥值定義
    else:
        #check_list = []  # 該行確認是否有已排序的類別

        if len(total_pos[i]) == cluster_num:  # 假如該行紀錄點數量剛好等於總分群數
            for j in total_pos[i]:
                value = place_to_value(j)
                if not clustered:  # 未歸類初始化，則直接定義參考值。
                    pre_color[cluster_count-1] = opening[j, i]
                    # pre_color[cluster_count-1] = lab_img[j, i]
                    pre_locate[cluster_count-1] = j
                    total_cluster[cluster_count-1].append(value)
                    print("已定義類別", cluster_count, "初始位置於", i, "行的", pre_locate[cluster_count-1], "座標")
                    cluster_count = cluster_count+1
                    locate = i
                else:  # 已歸類參考點。
                    pass
                    '''
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
                                '''

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
            for n in range(0, cluster_num):
                total_cluster[n].append("")
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
            for j in total_pos[i]:
                value = place_to_value(j)
                if not clustered:
                    pass
                else:
                    pass


def expect_locate(in_which_cluster, in_which_col):
    return result


#for i in range(locate+1, 241):
for i in range(locate+1, len(total_pos)):
    value = x_label_to_value(i)
    x_value.append(value)
    if total_pos[i] == [None]:
        for n in range(0, cluster_num):
            total_cluster[n].append("")
    else:
        check_list = []
        check_count = 0
        for j in total_pos[i]:
            dist_list = []
            color_dist_list = []
            value = place_to_value(j)
            for k in range(0, cluster_num):
                dist = abs(pre_locate[k] - j)
                color_dist = colordist(pre_color[k], opening[j, i])
                #color_dist = labdist(pre_color[k], lab_img[j, i])
                #print(color_dist)
                dist_list.append(dist)
                color_dist_list.append(color_dist)
            for a, element in enumerate(dist_list):
                if element < 20:
                    place = a
                    try:
                        if check_list.index(place):
                            print("有重疊的值，於座標(", j, i, ")")

                    except ValueError:
                        #if check_count >= len(total_pos[i]):

                        if color_dist_list[place] < 125:
                            check_list.append(place)
                            check_count = check_count+1
                            #pre_color[place] = opening[j, i]
                            #pre_color[place] = lab_img[j, i]
                            pre_locate[place] = j
                            total_cluster[place].append(value)
                            break
                        else:
                            print("未歸類的值，於座標(", j, i, ")", place)

        for l in range(0, cluster_num):
            try:
                check_list.index(l)
            except ValueError:
                total_cluster[l].append("")


sheet2 = book.add_sheet('sheet2')
for i, e in enumerate(x_value):
    sheet2.write(i, 0, e)

# for i in range(0, cluster_num):
#     correct_data(total_cluster[i])

for k in range(0, cluster_num):
    col = k+1
    for i, e in enumerate(total_cluster[k]):
        if type(e) != str:
            sheet2.write(i, col, e)
        else:
            sheet2.write(i, col, str(total_cluster[k][i]))
        # sheet2.write(i,k,str(total_cluster[k][i]))
name = "new_data.xls"
try:
    book.save(name)
    book.save(TemporaryFile())
except PermissionError:
    print("請關閉Excel後存檔")
    pass

