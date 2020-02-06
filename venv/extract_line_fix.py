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
        if i == len(cluster) - 1:
            break
        pre = cluster[i]
        if type(pre) != str:
            if type(cluster[i + 1]) == str:
                for j in range(i + 1, len(cluster)):
                    end = cluster[j]
                    if type(end) != str:
                        # print(i, pre)
                        # print(j, end)
                        d = (end - pre) / (j - i)
                        for k in range(i + 1, j):
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



# img = cv2.imread("tri_black_gridout.jpg")
img = cv2.imread("black_grid.jpg")
img = cv2.imread("C:/Users/Burny/PycharmProjects/remake/venv/output/Grid_removed.jpg")
# img = cv2.imread("blurop.jpg")
# img = cv2.imread("Grid_ removed_f.jpg")
# img = cv2.imread("Grid_removed1.jpg")
# img = cv2.imread("Grid_removed2.jpg")

thr_value = 80
thr1_value = 60
thr_place = 143
thr1_place = 215
x_label_value_1 = 1
x_label_value_2 = 2
x_label_place_1 = 54
x_label_place_2 = 160

[a, b, c] = np.shape(img)  # a=484 b=996,c=3
row = a
col = b
print("row=", row, "col=", col)

kernel = np.ones((7, 7), np.uint8)
cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
blur = cv2.blur(img, (3, 3))
opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)  # BGR
opening = cv2.dilate(opening, (3, 3))
lab_img = cv2.cvtColor(opening, cv2.COLOR_BGR2LAB)
hsv_img = cv2.cvtColor(opening, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(opening, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow("tt", thresh)
# cv2.waitKey()
cv2.imwrite("t.jpg", thresh)
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
total_cluster_pos = []
pre_locate = []
pre_locate_col = []
temp_locate = []
pre_color = []
x_value = []
for i in range(0, cluster_num):
    # x_value.append([])
    total_cluster.append([])
    total_cluster_pos.append([])
    pre_locate.append([""])
    pre_locate_col.append([""])
    temp_locate.append([""])
    pre_color.append([])


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


def labdist(lab_1, lab_2):
    l_1, a_1, b_1 = lab_1
    l_2, a_2, b_2 = lab_2
    return np.sqrt((l_1 - l_2) ** 2 + (a_1 - a_2) ** 2 + (b_1 - b_2) ** 2)


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
            total_cluster_pos[n].append("")
    # 第一次進入分群閥值定義
    else:
        if len(total_pos[i]) == cluster_num:  # 假如該行紀錄點數量剛好等於總分群數
            for j in total_pos[i]:
                value = place_to_value(j)
                if not clustered:  # 未歸類初始化，則直接定義參考值。
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


def expect_locate(in_which_cluster, in_which_col):
    ones = False
    count = 0
    for i in range(in_which_col, len(total_pos)):
        count = count+1
        if len(total_pos[i]) == cluster_num:
            if ones:
                ones = False
                break
            for j in total_pos[i]:
                dist = abs(pre_locate[in_which_cluster] - j)
                if dist < count*10 and not ones:
                    end = j
                    end_place = i
                    ones = True
                    # print(end, in_which_col - 1, "to", end_place, in_which_cluster)
                    # count = i - (in_which_col - 1)
                    break
    if count > 10:
        # print(in_which_col)
        return None
    for i in range(0, count):
        dist = []
        dist[:] = [abs(end - x) for x in total_pos[end_place]]
        place = dist.index(min(dist))  # 1
        end = total_pos[end_place][place]
        end_place = end_place - 1
    # print(end)
    return end

check_cross = False
count_none = 0
cross_num = 0
for i in range(locate + 1, len(total_pos)):
    value = x_label_to_value(i)
    x_value.append(value)
    # if 太多None再後半段 則視為數據中止
    if total_pos[i] == [None] and i > 4/5*len(total_pos):
        count_none = count_none+1
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
            check_close_list[:] = [abs(x-pre_locate[j]) for x in pre_locate]
            check_close_num = 0
            for k in check_close_list:
                if k < row/15:
                    check_close_num = check_close_num+1
                    if k == 0:
                        pass
                    else:
                        close_place = check_close_list.index(k)
            if check_close_num == 2:
                check_close = True
                close_place_1 = j  # 0
                # print("Close in ", i, close_place_1, close_place)
                break

        # check cross
        check_cross_num = 0
        if cross_num != 0 and len(total_pos[i]) >= cluster_num:
            cross_num = 0

        if len(total_pos[i]) < cluster_num and check_close:
            if check_cross:
                pass
            else:
                # check_cross = False
                if cross_num == 0:
                    pre_1 = pre_locate[close_place_1]  # 376
                    pre = pre_locate[close_place]  # 363

                    # print("first cross in", i, pre_1, pre)

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
                    # print("Error in ", i)
                    cross_num = 0
                else:
                    if pre_1 > pre:
                        for l in total_pos[i]:
                            if l in range(pre, pre_1):
                                check_cross_num = check_cross_num + 1
                                # check_slope = l
                    else:
                        for l in total_pos[i]:
                            if l in range(pre_1, pre):
                                check_cross_num = check_cross_num + 1
                            # check_slope = l
        if check_cross_num == 1:
            cross_num = cross_num + 1

            # if cross_slope > 0:
            #     cross_slope = 1
            #     cross_slope_1 = -1
            # else:
            #     cross_slope = -1
            #     cross_slope_1 = 1

        if cross_num >= 2:
            if pre > pre_1:
                # pre_1 = pre_locate[close_place_1]  # 363
                # pre = pre_locate[close_place]  # 376
                cross_slope = -1
                cross_slope_1 = 1
            else:
                cross_slope = 1
                cross_slope_1 = -1
            if abs(pre_locate_col[close_place_1] - pre_locate_col[close_place]) > col*0.1:
                pass
            else:
                check_cross = True
                cross_num = 0
                print("Cross in ", i, close_place, close_place_1)
        for j in total_pos[i]:
            if count_none > 10:
                break
            if len(total_pos[i]) <= cluster_num:
                #  特殊案例 確認交叉後 第一次歸類
                if check_cross and len(total_pos[i]) == cluster_num:
                    # pre_locate[close_place]  # 179
                    # cross_slope  # -
                    # pre_locate[close_place_1]  # 160
                    # cross_slope_1 # +
                    temp = []
                    tempp = []
                    check_place = []
                    temp[:] = [j - pre_locate[close_place] for j in total_pos[i]]
                    tempp[:] = [abs(j - pre_locate[close_place]) for j in total_pos[i]]  # 212 65 51
                    abs_temp = tempp.copy()
                    place = tempp.index(min(tempp))  # 2
                    pos = total_pos[i][place]  # 421
                    tempp.pop(place)
                    place1 = abs_temp.index(min(tempp))  # 1
                    pos1 = total_pos[i][place1]  # 407
                    # Error correct 如果 temp = [-3, 3, 211]，abs後則會有兩相同的值之誤判
                    if abs(pos - pos1) > row*0.15:
                        break
                    if place == place1:
                        place1 = temp.index(min(tempp))
                        pos1 = total_pos[i][place1]
                    if pos > pos1:
                        pos_slope = 1
                        pos1_slope = -1
                    else:
                        pos_slope = -1
                        pos1_slope = 1
                    value = place_to_value(pos)  # 421
                    value1 = place_to_value(pos1)  # 407
                    check_list.append(close_place)  # 2
                    check_list.append(close_place_1)  # 1
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





            #         if cross_slope < 0:  # 遞減
            #             expect = float("-inf")
            #             double_check = []
            #             # 找斜率為負 統計有可能的值
            #             for j in temp:
            #                 if j < 0:
            #                     if expect < j:
            #                         expect = j
            #                         double_check.append(j)
            #             if len(double_check) == 1:  # =1表示為只有一個可能
            #                 check_cross_place = temp.index(double_check[0])
            #                 check_cross_place_1 = check_cross_place + 1
            #                 value = place_to_value(total_pos[i][check_cross_place])
            #                 value1 = place_to_value(total_pos[i][check_cross_place_1])
            #                 check_list.append(check_cross_place)
            #                 check_list.append(check_cross_place_1)
            #                 temp_locate[close_place] = total_pos[i][check_cross_place]
            #                 temp_locate[close_place_1] = total_pos[i][check_cross_place_1]
            #                 total_cluster[close_place].append(value)
            #                 total_cluster[close_place_1].append(value1)
            #                 total_cluster_pos[close_place].append(total_pos[i][check_cross_place])
            #                 total_cluster_pos[close_place_1].append(total_pos[i][check_cross_place_1])
            #             else:  # 複數可能則需判斷何者正確
            #                 find_min = []
            #                 for j in double_check:
            #                     check_cross_place = temp.index(j)
            #                     check_cross_place_1 = check_cross_place + 1
            #                     if check_cross_place_1 >= len(pre_locate):
            #                         break
            #                     # 另一必須為遞增
            #                     dist = total_pos[i][check_cross_place_1] - pre_locate[close_place_1]
            #                     find_min.append(dist)
            #                     if dist < 0:
            #                         pass
            #                     else:
            #                         if abs(dist) < 40:
            #                             value = place_to_value(total_pos[i][check_cross_place])
            #                             value1 = place_to_value(total_pos[i][check_cross_place_1])
            #                             check_list.append(check_cross_place)
            #                             check_list.append(check_cross_place_1)
            #                             temp_locate[close_place] = total_pos[i][check_cross_place]
            #                             temp_locate[close_place_1] = total_pos[i][check_cross_place_1]
            #                             total_cluster[close_place].append(value)
            #                             total_cluster[close_place_1].append(value1)
            #                             total_cluster_pos[close_place].append(total_pos[i][check_cross_place])
            #                             total_cluster_pos[close_place_1].append(total_pos[i][check_cross_place_1])
            #
            #             for j in total_pos[i]:
            #                 dist_list = []
            #                 value = place_to_value(j)
            #                 for k in range(0, cluster_num):
            #                     dist = abs(pre_locate[k] - j)
            #                     dist_list.append(dist)
            #                 for a, element in enumerate(dist_list):
            #                     if dist_list.index(min(dist_list)) == a:
            #                         place = a
            #                         try:
            #                             try_y = check_list.index(place)
            #                         except ValueError:
            #                             check_list.append(place)
            #                             check_count = check_count + 1
            #                             temp_locate[place] = j
            #                             total_cluster[place].append(value)
            #                             total_cluster_pos[place].append(j)
            #         else:  # 遞增
            #             expect = float("inf")
            #             double_check = []
            #             # 找斜率為正 統計有可能的值
            #             for j in temp:
            #                 if j > 0:
            #                     double_check.append(j)
            #                     if expect > j:
            #                         expect = j
            #             if len(double_check) == 1:  # =1表示為只有一個可能
            #                 check_cross_place = temp.index(double_check[0])
            #                 check_cross_place_1 = check_cross_place - 1
            #                 value = place_to_value(total_pos[i][check_cross_place])
            #                 value1 = place_to_value(total_pos[i][check_cross_place_1])
            #                 check_list.append(check_cross_place)
            #                 check_list.append(check_cross_place_1)
            #                 temp_locate[close_place] = total_pos[i][check_cross_place]
            #                 temp_locate[close_place_1] = total_pos[i][check_cross_place_1]
            #                 total_cluster[close_place].append(value)
            #                 total_cluster[close_place_1].append(value1)
            #                 total_cluster_pos[close_place].append(total_pos[i][check_cross_place])
            #                 total_cluster_pos[close_place_1].append(total_pos[i][check_cross_place_1])
            #             else:  # 複數可能則需判斷何者正確
            #                 for j in double_check:
            #                     check_cross_place = temp.index(j)
            #                     check_cross_place_1 = check_cross_place - 1
            #                     # 另一必須為遞減
            #                     dist = total_pos[i][check_cross_place_1] - pre_locate[close_place_1]
            #                     if dist > 0:
            #                         pass
            #                     else:
            #                         if abs(dist) < 40:
            #                             value = place_to_value(total_pos[i][check_cross_place])
            #                             value1 = place_to_value(total_pos[i][check_cross_place_1])
            #                             check_list.append(check_cross_place)
            #                             check_list.append(check_cross_place_1)
            #                             temp_locate[close_place] = total_pos[i][check_cross_place]
            #                             temp_locate[close_place_1] = total_pos[i][check_cross_place_1]
            #                             total_cluster[close_place].append(value)
            #                             total_cluster[close_place_1].append(value1)
            #                             total_cluster_pos[close_place].append(total_pos[i][check_cross_place])
            #                             total_cluster_pos[close_place_1].append(total_pos[i][check_cross_place_1])
            #
            #             for j in total_pos[i]:
            #                 dist_list = []
            #                 value = place_to_value(j)
            #                 for k in range(0, cluster_num):
            #                     dist = abs(pre_locate[k] - j)
            #                     dist_list.append(dist)
            #                 for a, element in enumerate(dist_list):
            #                     if dist_list.index(min(dist_list)) == a:
            #                         place = a
            #                         try:
            #                             try_y = check_list.index(place)
            #                         except ValueError:
            #                             check_list.append(place)
            #                             check_count = check_count + 1
            #                             temp_locate[place] = j
            #                             total_cluster[place].append(value)
            #                             total_cluster_pos[place].append(j)
            #         print("fff", i)
            #         check_cross = False
            #         break
            # else:
            #     break
            # if not cross need to expect cross place
            dist_list = []
            value = place_to_value(j)
            for k in range(0, cluster_num):
                dist = abs(pre_locate[k] - j)
                dist_list.append(dist)
            for a, element in enumerate(dist_list):
                if dist_list.index(min(dist_list)) == a:
                    if min(dist_list) > row/30:
                    #     print(pre_locate)
                    #     print("dist>20",i, j)
                        break
                    place = a
                    try:
                        try_y = check_list.index(place)
                        # fix = expect_locate(place, i + 1)
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

sheet2 = book.add_sheet('sheet2')
for i, e in enumerate(x_value):
    sheet2.write(i, 0, e)

for i in range(0, cluster_num):
    correct_data(total_cluster[i])
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

print("finish")
