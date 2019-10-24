import cv2
import numpy as np
from matplotlib import pyplot as plt


def line_detect(image):
    edges = cv2.Canny(image, 50, 150)
    lines = cv2.HoughLinesP(edges, 0.01, np.pi/180, 25, 25)
    lin_num = np.size(lines, 0)
    for i in range(lin_num):
        for x1, y1, x2, y2 in lines[i]:
            cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 3)
    return image


def find_bound(list, startline):
    startline = startline+1
    [bound] = np.shape(list)
    for i in range(startline, bound):
        if list[i] != 0:
            bound_1 = i
            break
        if i == bound-1:
            return None, None
    for i in range(bound_1, bound):
        if list[i] == 0:
            bound_2 = i
            break
    #print(bound_1)
    #print(bound_2)
    return bound_1, bound_2


def find_bound_inv(list, startline):
    startline = startline-1
    [bound] = np.shape(list)
    for i in range(startline, 0):
        if list[i] != 0:
            bound_1 = i
            break
        if i == bound-1:
            return None, None
    for i in range(bound_1, 0):
        if list[i] == 0:
            bound_2 = i
            break
    #print(bound_1)
    #print(bound_2)
    return bound_1, bound_2

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


img = cv2.imread('test.jpg')
test = cv2.imread('test.jpg')
[row, col, x] = np.shape(img)
result_img = np.zeros((row, col), dtype=np.uint8)
result_img = cv2.imread('test.jpg', 0)
ret, result_img = cv2.threshold(result_img, 240, 255, cv2.THRESH_OTSU)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
kernel = np.uint8([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
eroded = cv2.erode(result_img, kernel)
roi_in_img = []
[totallist, x] = cal_each_x_accumulation(eroded)
[x1, x2] = find_bound(totallist, 0)


plt.bar(x,totallist,1,color='b')
plt.xlabel('X座標',color='g')
plt.ylabel('每條直線',color='g')
plt.show()

# 提取長條ROI
vert_roi = eroded[:, x1:x2]
[row,col]=np.shape(vert_roi)
cv2.imshow("",vert_roi)
cv2.waitKey()
# 計算每一列之累積值
[totallist, xa] = cal_each_y_accumulation(vert_roi)
# 找ROI_1 放大並處理
[y1, y2] = find_bound(totallist, 0)
roi_in_img.append(y1)
roi_in_img.append(y2)
cv2.rectangle(test, (x1,y1), (x2, y2), (255, 255, 0),1)
roi_1 = img[y1:y2, x1:x2, :]
temp_a=(x2-x1)*5
temp_b=(y2-y1)*5
roi_1 = cv2.resize(roi_1, (temp_a,temp_b))
roi_1 = cv2.cvtColor(roi_1, cv2.COLOR_BGR2GRAY)
ret, roi_1 = cv2.threshold(roi_1, 170, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
roi_1 = cv2.dilate(roi_1, kernel)
# 找ROI_2 放大並處理
[y1, y2] = find_bound(totallist, y2)
roi_in_img.append(y1)
roi_in_img.append(y2)
cv2.rectangle(test, (x1,y1), (x2, y2), (255, 255, 0),1)
roi_2 = img[y1:y2, x1:x2, :]
temp_a=(x2-x1)*5
temp_b=(y2-y1)*5
roi_2 = cv2.resize(roi_2, (temp_a,temp_b))
roi_2 = cv2.cvtColor(roi_2, cv2.COLOR_BGR2GRAY)
ret, roi_2 = cv2.threshold(roi_2, 170, 255, cv2.THRESH_BINARY)
roi_2 = cv2.dilate(roi_2, kernel)

# 找ROI_3 放大並處理
[y1, y2] = find_bound(totallist, y2)
roi_in_img.append(y1)
roi_in_img.append(y2)
cv2.rectangle(test, (x1,y1), (x2, y2), (255, 255, 0),1)
roi_3 = img[y1:y2, x1:x2, :]
temp_a=(x2-x1)*5
temp_b=(y2-y1)*5
roi_3 = cv2.resize(roi_3, (temp_a,temp_b))
roi_3 = cv2.cvtColor(roi_3, cv2.COLOR_BGR2GRAY)
ret, roi_3 = cv2.threshold(roi_3, 170, 255, cv2.THRESH_BINARY)
roi_3 = cv2.dilate(roi_3, kernel)



cv2.imshow("roi_1", roi_1)
cv2.imshow("roi_2", roi_2)
cv2.imshow("roi_3", roi_3)

cv2.imwrite("roi_1.jpg",roi_1)
cv2.imwrite("roi_2.jpg",roi_2)
cv2.imwrite("roi_3.jpg",roi_3)
# K-NN Digit_Recognize
samples = np.loadtxt('generalsamples.txt', np.float32)
responses = np.loadtxt('generalresponses.txt', np.float32)
responses = responses.reshape((responses.size, 1))
model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)

roi1_x1 = []
roi1_x2 = []
[roi1_x1, roi1_x2] = find_roi_bound(roi_1)
[bound] = np.shape(roi1_x1)
ret, thresh = cv2.threshold(roi_1, 170, 255, cv2.THRESH_BINARY_INV)
#thresh = cv2.adaptiveThreshold(roi_1, 255, 1, 1, 11, 2)
roi_1 = cv2.cvtColor(roi_1, cv2.COLOR_GRAY2BGR)
out = np.zeros(roi_1.shape, np.uint8)
y1 = roi_in_img[0]
y2 = roi_in_img[1]
t2 = x2
[roi_row, roi_col, x] = np.shape(roi_1)
print("roi1=")
for i in range(bound):
    if roi1_x1[i] == None:
        break
    else:
        roi = thresh[0:roi_row, roi1_x1[i]:roi1_x2[i]]
        roismall = cv2.resize(roi, (10, 10))
        roismall = roismall.reshape((1, 100))
        roismall = np.float32(roismall)
        retval, results, neigh_resp, dists = model.findNearest(roismall, k=3)
        string = str(chr(int((results[0][0]))))
        cv2.putText(test, string, (t2, y1), 0, 0.5, (0, 0, 255))
        cv2.putText(out, string, (roi1_x1[i], round(roi_row/2)), 0, 1, (255, 255, 0))
        print(string)
        t2 = t2+2*x1


roi1_x1 = []
roi1_x2 = []
[roi1_x1, roi1_x2] = find_roi_bound(roi_2)
[bound] = np.shape(roi1_x1)
ret, thresh = cv2.threshold(roi_2, 170, 255, cv2.THRESH_BINARY_INV)
#thresh = cv2.adaptiveThreshold(roi_2, 255, 1, 1, 11, 2)
roi_2 = cv2.cvtColor(roi_2, cv2.COLOR_GRAY2BGR)
out = np.zeros(roi_2.shape, np.uint8)
y1 = roi_in_img[2]
y2 = roi_in_img[3]
t2 = x2
[roi_row, roi_col, x] = np.shape(roi_2)
print("roi2=")
for i in range(bound):
    if roi1_x1[i] == None:
        break
    else:
        roitest=roi_2[0:roi_row, roi1_x1[i]:roi1_x2[i]]
        roi = thresh[0:roi_row, roi1_x1[i]:roi1_x2[i]]
        roismall = cv2.resize(roi, (10, 10))
        roismall = roismall.reshape((1, 100))
        roismall = np.float32(roismall)
        retval, results, neigh_resp, dists = model.findNearest(roismall, k=3)
        string = str(chr(int((results[0][0]))))
        cv2.putText(test, string, (t2, y1), 0, 0.5, (0, 0, 255))
        cv2.putText(out, string, (roi1_x1[i], round(roi_row/2)), 0, 1, (255, 255, 0))
        print(string)
        t2 = t2+2*x1


roi1_x1 = []
roi1_x2 = []
[roi1_x1, roi1_x2] = find_roi_bound(roi_3)
[bound] = np.shape(roi1_x1)
ret, thresh = cv2.threshold(roi_3, 180, 255, cv2.THRESH_BINARY_INV)
#thresh = cv2.adaptiveThreshold(roi_3, 255, 1, 1, 11, 2)
roi_3 = cv2.cvtColor(roi_3, cv2.COLOR_GRAY2BGR)
out = np.zeros(roi_3.shape, np.uint8)
y1 = roi_in_img[4]
y2 = roi_in_img[5]
t2 = x2
[roi_row, roi_col, x] = np.shape(roi_3)
print("roi3=")
for i in range(bound):
    if roi1_x1[i] == None:
        break
    else:
        roi = thresh[0:roi_row, roi1_x1[i]:roi1_x2[i]]
        roismall = cv2.resize(roi, (10, 10))
        roismall = roismall.reshape((1, 100))
        roismall = np.float32(roismall)
        retval, results, neigh_resp, dists = model.findNearest(roismall, k=3)
        string = str(chr(int((results[0][0]))))
        cv2.putText(test, string, (t2, y1), 0, 0.5, (0, 0, 255))
        cv2.putText(out, string, (roi1_x1[i], round(roi_row/2)), 0, 1, (255, 255, 0))
        print(string)
        t2 = t2+2*x1

cv2.imshow("", test)
cv2.waitKey()




