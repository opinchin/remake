import os
import cv2
import numpy as np
def dataregion_detect(img):
    try:
        # print("img", img)
        origin = cv2.imread(img)
        gray = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
        ret, gray = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
        [a, b] = np.shape(gray)
        image_data = origin.copy()
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
            print("找不到右邊界")

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
            print("找不到上邊界")
        downbound = target
        cv2.imwrite(os.path.join(save_path, str(img)+'data_region.jpg'), image_data)
    except:
        print("Error in", img)




path = 'C:/Users/Burny/PycharmProjects/remake/venv/all_origin'
save_path='C:/Users/Burny/PycharmProjects/remake/venv/all_origin/dataregion'
os.chdir(path)
allfile = os.listdir(path)
name = 100
# for i in allfile:
#     fname = os.path.splitext(i)[0]  # 分解出當前的檔案路徑名字
#     ftype = os.path.splitext(i)[1]  # 分解出當前的副檔名
#     print(ftype)
#     os.rename(i, str(name)+ftype)
#     name = name+1
for i in allfile:
    # print(os.getcwd())
    dataregion_detect(i)
    print(i)
