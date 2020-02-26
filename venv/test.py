import tkinter
import cv2
import PIL.Image, PIL.ImageTk
from easygui import fileopenbox
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema
import pickle
import xlwt
from tempfile import TemporaryFile
import Gui_define
'''
# Read the image and create a blank mask
img = cv2.imread('1.jpg')
h,w = img.shape[:2]
mask = np.zeros((h,w), np.uint8)

# Transform to gray colorspace and threshold the image
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Perform opening on the thresholded image (erosion followed by dilation)
kernel = np.ones((2,2),np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

cv2.imshow("",thresh);
cv2.waitKey()
# Search for contours and select the biggest one and draw it on mask
_, contours, hierarchy = cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
cnt = max(contours, key=cv2.contourArea)
cv2.drawContours(mask, [cnt], 0, 255, -1)

# Perform a bitwise operation
res = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow("",mask);
cv2.waitKey()
# Threshold the image again
gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Find all non white pixels
non_zero = cv2.findNonZero(thresh)

# Transform all other pixels in non_white to white
for i in range(0, len(non_zero)):
    first_x = non_zero[i][0][0]
    first_y = non_zero[i][0][1]
    first = res[first_y, first_x]
    res[first_y, first_x] = 255

# Display the image
cv2.imshow('img', res)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
'''
inputImage = cv2.imread("1.jpg")
inputImageGray = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(inputImageGray,100,200,apertureSize = 3)
minLineLength = 1
maxLineGap = 2
lines = cv2.HoughLinesP(edges,cv2.HOUGH_PROBABILISTIC,5*np.pi/180, 30, minLineLength,maxLineGap,20)
for x in range(0, len(lines)):
    for x1,y1,x2,y2 in lines[x]:
       # cv2.line(inputImage,(x1,y1),(x2,y2),(0,128,0),2, cv2.LINE_AA)
        pts = np.array([[x1, y1 ], [x2 , y2]], np.int32)
        cv2.polylines(inputImage, [pts], True, (0,255,0))

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(inputImage,"Tracks Detected", (500, 250), font, 0.5, 255)
cv2.imshow("Trolley_Problem_Result", inputImage)
cv2.imshow('edge', edges)
cv2.waitKey(0)
'''
'''
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# Perform opening on the thresholded image (erosion followed by dilation)
kernel = np.ones((2,2),np.uint8)
img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
cv2.imshow("",img)
cv2.waitKey()
'''
'''
img=cv2.imread('1.jpg')
Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 20
ret, label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image




x=np.size(label)

b=[255,255,255]
c=[0,0,0]

center=np.row_stack((center,c))
center=np.row_stack((center,c))
center= np.uint8(center)
y=label.copy()
for j in range(K):
    for i in range(x):
        if label[i]==j:
            y[i]=j
        else:
            y[i]=K
    print(j)
    res = center[y.flatten()]

    res2 = res.reshape((img.shape))
    cv2.imshow('res2', res2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''
"""
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, thin,opening,square
from skimage import io,measure
from skimage.color import rgb2gray
import cv2
original = io.imread('1.jpg')
image = rgb2gray(original)
from skimage.filters import threshold_otsu
Otsu_Threshold = threshold_otsu(image)
image = image < Otsu_Threshold

contours = measure.find_contours(image, 0.8,'high','low')

# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(image, interpolation='nearest', cmap=plt.cm.gray)

for n, contour in enumerate(contours):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()
#image = opening(image,square(3))
skeleton = skeletonize(image)
thinned = thin(image)
thinned_partial = thin(image, max_iter=25)

fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_title('original')
ax[0].axis('off')

ax[1].imshow(skeleton, cmap=plt.cm.gray, interpolation='nearest')
ax[1].set_title('skeleton')
ax[1].axis('off')

ax[2].imshow(thinned, cmap=plt.cm.gray, interpolation='nearest')
ax[2].set_title('thinned')
ax[2].axis('off')

ax[3].imshow(thinned_partial, cmap=plt.cm.gray, interpolation='nearest')
ax[3].set_title('partially thinned')
ax[3].axis('off')

fig.tight_layout()
plt.show()
"""
'''
import numpy as np
import cv2

img = cv2.imread('4.JPG')
img2 = img.copy()
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret2,th2 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
[data_row, data_col, x]= np.shape(img)
for i in range(data_row):
    for j in range(data_col):
        if th2[i, j] == 255:
            img[i, j, :] = 255
edges = cv2.Canny(gray,50,150)
cv2.imshow("",hsv)
cv2.waitKey()
cv2.destroyAllWindows()
minLineLength=100
lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=150,lines=np.array([]), minLineLength=minLineLength,maxLineGap=10)

a,b,c = lines.shape


for i in range(a):
    if lines[i][0][1] == lines[i][0][3]:  #Y同 水平線
        y=lines[i][0][1]
        x1=lines[i][0][0]
        x2=lines[i][0][2]
        temp = gray[y, x1:x2]
        max = np.argmax(np.bincount(temp)) #取灰階最多的值

        if max ==255: #不取白線 若為白線則再取第二大的灰階值
            temp= np.delete(temp, np.where(temp == 255))
            max = np.argmax(np.bincount(temp))  # 取灰階最多的值
        for j in range(x1, x2):
            if gray[y, j] == max:
                cv2.line(img, (j, y), (j, y), (255, 255, 255), 2, cv2.LINE_AA)
            else:
                pass

    elif lines[i][0][0] == lines[i][0][2]:#X同 垂直線
        x=lines[i][0][0]
        y1=lines[i][0][1]
        y2=lines[i][0][3]
        temp = gray[y2:y1, x]
        max = np.argmax(np.bincount(temp))  # 取灰階最多的值

        if max == 255:  # 不取白線 若為白線則再取第二大的灰階值
            temp = np.delete(temp, np.where(temp == 255))
            max = np.argmax(np.bincount(temp))  # 取灰階最多的值
        for j in range(y2, y1):
            if gray[j, x] == max:
                cv2.line(img, (x, j), (x, j), (255, 255, 255), 2, cv2.LINE_8)
            else:
                pass

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret2, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("",th2)
cv2.waitKey()
cv2.destroyAllWindows()
a,contours, hierarchy = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    size = cv2.contourArea(cnt)
    if size>1000:
        cv2.drawContours(img2, [cnt], 0, (255, 255, 0), 3)
cv2.imshow("",img)
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imshow("",img2)
cv2.waitKey()
cv2.destroyAllWindows()
'''
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema

img = cv2.imread('33.jpg')
cv2.imshow("Origin",img)
cv2.waitKey()
img=img[:,:,[2,1,0]]
hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
[a,b,c]=np.shape(hsv)
m_col = round(b/2)

#撇除黑灰白的顏色直方圖

lower_white =np.array([0, 50, 50])
upper_white =np.array([180, 255, 255])
mask = cv2.inRange(hsv,lower_white,upper_white)

hsv_count=[]
for i in range(0,a):
    for j in range(0,b):
        if mask[i,j] != 0:
            value = hsv[i,j,0]
            hsv_count.append(value)



z=plt.hist(hsv_count,256,[0,256])
zz=z[0]
text1="Accumulation of the H channel except for Black & White"
plt.title(text1)
plt.show()
#取HSV顏色峰值

h=len(zz)
num=[]

for i in range(0,h):
    if zz[i]<100:
        zz[i]=0
    num.append(i)

#plt.plot(num,zz,'k')

k=argrelextrema(zz, np.greater,order=5)
[_ ,k_num]=np.shape(k)

high_hsv=[]
thr=5

#x = np.int(k[0][0])
#檢查邊界問題
for i in range(0,k_num):
    a=int(k[0][i])
    if a+thr>179:#a=178
        list=[]
        list.append(a)
        for i in range(1,6):
            if a+i>179:#a=178，i=2，須得到0
                b=a+i-180
                list.append(b)
            else:
                b=a+i
                list.append(b)
        list1=zz[a],zz[list[1]],zz[list[2]],zz[list[3]],zz[list[4]],zz[list[5]]
        a=np.where(list1==max(list1))
        a = np.int(a[0][0])
        a=list[a]
    elif a-thr<0:#a=1
        list=[]
        list.append(a)
        for i in range(1,6):
            if a-i<0:#a=1 , i=2,須得到179
                b=180+a-i
                list.append(b)
            else:
                b=a-i
                list.append(b)
        list1 = zz[a], zz[list[1]], zz[list[2]], zz[list[3]], zz[list[4]], zz[list[5]]
        a=np.where(list1==max(list1))
        a=np.int(a[0][0])
        a=list[a]
    high_hsv.append(a)
#順序排列H通道峰值
high_hsv=sorted(set(high_hsv))
[k_num]=np.shape(high_hsv)
print("峰值共",k_num,"個")


for i in high_hsv:
    x=i
    text1 = 'Origin'
    text2 = 'When H ='+np.str(i)+'+-'+np.str(thr)
    print("閥值=",i)
    if x+thr>179:#x=176
        lower=x-thr,50,50
        upper=x+thr,255,255#176+5=181
        carry_upper=(x+thr-179,255,255)
        mask1 = cv2.inRange(hsv, lower, upper)
        mask2 = cv2.inRange(hsv, (0,50,50) ,carry_upper)
        mask = mask1 + mask2
    elif x-thr<0:#x=0
        lower=(0,50,50)
        upper=(x+thr,255,255)
        carry_lower=(180-(thr-x),50,50)
        mask1 = cv2.inRange(hsv, lower, upper)
        mask2 = cv2.inRange(hsv, carry_lower, (180,255,255))
        mask = mask1 + mask2
    else:
        lower = (x-thr,50,50)
        upper = (x+thr,255,255)
        mask = cv2.inRange(hsv,lower, upper)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img,img, mask= mask)
    plt.figure(1)
    plt.subplot(1 ,2 ,1)
    plt.imshow(img)
    plt.title(text1)
    plt.subplot(1 ,2 ,2)
    plt.imshow(res)
    plt.title(text2)
    plt.show()
   # plt.waitforbuttonpress

'''
'''
for i in range(0,a):
    for j in  range(0,b):
        if abs(hsv[i,j,0])<=100:
            img[i,j,:]=[255,255,255]
        else:
            img[i,j,:]=[0,0,0]

cv2.imshow("img",img)
cv2.waitKey()
cv2.destroyAllWindows()
'''
#
# ''''''
# #滑鼠點pixel 比較相鄰的pixel距離
# '''
# import cv2
# import numpy as np
#
# # mouse callback function
#
# def select_origin(event, x, y, flags, param):
#     global coordinate
#     if event == cv2.EVENT_LBUTTONDOWN:
#         coordinate=(y,x)
#         print(img[y,x])
#         print(y,x)
#         imgok=img[y - 1:y + 1, x - 1:x + 1, :]
#         imgok = cv2.resize(imgok, (300, 300))
#         cv2.line(imgok, (140, 150), (160, 150), (0, 0, 255), 1)
#         cv2.line(imgok, (150, 140), (150, 160), (0, 0, 255), 1)
#         cv2.imshow("123", imgok)
#
#     if event ==cv2.EVENT_MOUSEMOVE:
#         try:
#             imgk = img[y - 5:y + 5, x - 5:x + 5, :]
#             imgk = cv2.resize(imgk, (300, 300))
#             cv2.line(imgk, (140, 150), (160, 150), (0, 0, 255), 1)
#             cv2.line(imgk, (150, 140), (150, 160), (0, 0, 255), 1)
#             cv2.imshow("", imgk)
#         except:
#             pass
#
# # 创建图像与窗口并将窗口与回调函数绑定
# img = cv2.imread("4.jpg")
# img= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# coordinate=1
# cv2.namedWindow('image')
#
# while coordinate==1:
#     print("選取後按任意鍵確定")
#     cv2.setMouseCallback('image', select_origin)
#     cv2.imshow('image', img)
#     cv2.waitKey()
#     if coordinate!=1:
#         cv2.destroyAllWindows()
# print("您選取的座標為",coordinate)
#
# def tracing_line(img,coordinate):
#     #y軸
#     ny1=coordinate[0]
#     ny2=coordinate[0]+1
#     ny3=coordinate[0]-1
#     #x軸
#     nx=coordinate[1]+1
#   #  p1=int(img[coordinate[0],coordinate[1],0])
#     p1_hsv= img[coordinate[0],coordinate[1]]
#     p1_hsv=p1_hsv.astype(float)
#     p2_hsv_1=img[ny1,nx]
#     p2_hsv_2=img[ny2,nx]
#     p2_hsv_3=img[ny3,nx]
#     p2_hsv_1 = p2_hsv_1.astype(float)
#     p2_hsv_2 = p2_hsv_2.astype(float)
#     p2_hsv_3 = p2_hsv_3.astype(float)
#     print(p2_hsv_1)
#     print(p2_hsv_2)
#     print(p2_hsv_3)
#     h_dist_1=min(abs(p1_hsv[0]-p2_hsv_1[0]),180-abs(p1_hsv[0]-p2_hsv_1[0]))
#     h_dist_2=min(abs(p1_hsv[0]-p2_hsv_2[0]),180-abs(p1_hsv[0]-p2_hsv_2[0]))
#     h_dist_3=min(abs(p1_hsv[0]-p2_hsv_3[0]),180-abs(p1_hsv[0]-p2_hsv_3[0]))
#     print(h_dist_1)
#     print(h_dist_2)
#     print(h_dist_3)
#     s_dist_1 = abs(p1_hsv[1] - p2_hsv_1[1])
#     s_dist_2 = abs(p1_hsv[1] - p2_hsv_2[1])
#     s_dist_3 = abs(p1_hsv[1] - p2_hsv_3[1])
#     print(s_dist_1)
#     print(s_dist_2)
#     print(s_dist_3)
#     v_dist_1 = abs(p1_hsv[2] - p2_hsv_1[2])
#     v_dist_2 = abs(p1_hsv[2] - p2_hsv_2[2])
#     v_dist_3 = abs(p1_hsv[2] - p2_hsv_3[2])
#     print(v_dist_1)
#     print(v_dist_2)
#     print(v_dist_3)
#     hsv_dist_1 = np.sqrt(h_dist_1 * h_dist_1 + s_dist_1 * s_dist_1 + v_dist_1 * v_dist_1)
#     hsv_dist_2 = np.sqrt(h_dist_2 * h_dist_2 + s_dist_2 * s_dist_2 + v_dist_2 * v_dist_2)
#     hsv_dist_3 = np.sqrt(h_dist_3 * h_dist_3 + s_dist_3 * s_dist_3 + v_dist_3 * v_dist_3)
#     print(hsv_dist_1)
#     print(hsv_dist_2)
#     print(hsv_dist_3)
#
#     img_s = img[coordinate[0] - 1:coordinate[0] + 1, coordinate[1] - 1:coordinate[1] + 1, :]
#     img_s = cv2.resize(img_s, (300, 300))
#     cv2.imshow("123", img_s)
#
#     img1 = img[ny1 - 1:ny1 + 1, nx - 1:nx + 1, :]
#     img1 = cv2.resize(img1, (300, 300))
#     cv2.imshow("1", img1)
#
#     img2 = img[ny2 - 1:ny2 + 1, nx - 1:nx + 1, :]
#     img2 = cv2.resize(img2, (300, 300))
#     cv2.imshow("2", img2)
#
#     img3 = img[ny3 - 1:ny3 + 1, nx - 1:nx + 1, :]
#     img3 = cv2.resize(img3, (300, 300))
#     cv2.imshow("3", img3)
#     cv2.waitKey()
#
#
# tracing_line(img,coordinate)
# '''
# #分群
# img = cv2.imread("Grid_removed (2).jpg")
# image = img.copy()
# hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# #img = cv2.medianBlur(img,3)
# [a,b,c] = np.shape(img)#a=420 b=959,c=3
# #img = cv2.medianBlur(img,3)
# gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# #gray=cv2.blur(gray,(3,3))
# #gray=cv2.GaussianBlur(gray,(3,3),2)
# #gray=cv2.medianBlur(gray,3)
#
# kernel = np.ones((5,5),np.uint8)
# gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
#
# ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
# # cv2.imwrite("THR.jpg", thresh)
# # 各行的紀錄點位置
# total_pos = []
# for i in range(0, b):
#     pos = []  # 紀錄該行的pixel值
#     for j in range(0, a):  # 每一列
#         pos.append(thresh[j, i])
#     add_none = []  # 欲添加於Total_pos 之 List
#     try:
#         [k1, k2, k3, k4] = Gui_define.find_total_bound(pos)
#         temp = [round((k1[i]+k2[i]-1)/2) for i in range(len(k1))]
#         total_pos.append(temp)
#         for k in temp:
#             print(k)
#             image[k-10:k+10, i, :] = 50
#     except UnboundLocalError:
#         total_pos.append(add_none)
# cv2.imshow("00", image)
# cv2.waitKey()
# # 若該行紀錄點為空集合，則將其補上None
# for i in range(0, np.size(total_pos)):
#     if not total_pos[i]:
#         total_pos[i] = [None]
# # 將每一行位置儲存至Excel分頁一
# '''
# cv2.imshow("Origin Picture",thresh1)
# #cv2.imwrite("ggg.jpg",thresh1)
# cv2.waitKey()
#
# #各行的紀錄點
# total_point=[]
# for g in range(0,b):# 每一行
#     lists = np.zeros(a)# 紀錄該行的分群狀況
#     lists2 = [] # 紀錄該行的pixel值
#     for h in range(0,a):# 每一列
#         lists2.append(thresh1[h,g])
#         cluster = 0
#         start = 0
#     lists2=list(map(int,lists2))
#     for i in range(0,a):
#         if i ==0:
#             #當i=0，設定pre為第一個pixel的值，以便後面的比較
#             pre=lists2[0]
#         else:
#             #當繼續往下查找，設定now為下一個值，並與pre比較大小，若相同則差異不大，反之則開始進行分群
#             now=lists2[i]
#             temp = i - 1
#             dist=abs(pre-now)#dist是為上方pixel 與 下方pixel的差異
#             if dist>50:#如果差異過大，則將上方所有pixel歸為一類，直到將該行所有pixel分類完成。
#                 if start==temp:
#                     lists[start]=cluster
#                 else:
#                     for j in range(start,i):
#                         lists[j]=cluster
#                     start=i
#                     pre=now
#                     cluster=cluster+1
#             elif i ==len(lists)-1:#End
#                 for j in range(start,len(lists)):
#                     lists[j]=cluster
#             else:
#                 pre=now
#         point = []
#     #確認各行紀錄點
#     for i in range(0,np.int(max(lists))+1):
#         #規則一:每一行太過於長的分群 與線段性質不吻合 忽略
#              #若吻合則記錄該分群之中心點當作準紀錄點
#         long = np.size(np.where(lists==i))
#         if long < 30:
#             range_=np.where(lists==i)
#             mid = round((long+1)/2)-1
#             result_point=range_[0][mid]
#             #規則二:準記錄點不得等於白色
#             if thresh1[result_point,g]==255:
#                 pass
#             else:
#                 point.append(range_[0][mid])
#            # print(thresh1[mid][g])
#     total_point.append(point)
# #若該行紀錄點為空集合，則將其補上None
# for i in range(0, np.size(total_point)):
#     if total_point[i] == []:
#         total_point[i] = [None]
# '''
# book = xlwt.Workbook()
# sheet1 = book.add_sheet('sheet1')
# for i, e in enumerate(total_pos):
#     sheet1.write(i, 0, str(total_pos[i]))
# name = "new_data.xls"
# book.save(name)
# book.save(TemporaryFile())
#
#
# cluster_num=0
# pre_cluster_num=0
# #取出所有記錄點結果之數量的最大值
# for i in range(len(total_pos)):
#     a=len(total_pos[i])
#     if a>cluster_num:
#         count=0
#         for j in range(len(total_pos)):
#            # a = len(total_point[j])
#             if a == len(total_pos[j]):
#                 count = count + 1
#         if count < len(total_pos) / 10:
#             cluster_num=pre_cluster_num
#         else:
#             pre_cluster_num=a
#             cluster_num=a
# print("分成", cluster_num, "類")
#
# # 分群對應值 與 分群參考點 之初始化
# total_cluster = []
# pre_locate = []
# for i in range(0,cluster_num):
#     total_cluster.append([])
#     pre_locate.append([""])
# # 將每一行記錄點再次分群，目的是將每一條線條歸類為獨立分群，是為分群對應值，並且對應圖表上的原始資料，可作圖
# # 3/22 成功分群
# cluster_count=0
# clustered=False
# gg=0
# gg1=0
# thr_value=110
# thr1_value=50
# thr_place=41
# thr1_place=281
# for i in range(len(total_pos)):
#     #空集合不分群
#     if total_pos[i]==[None]:
#         for n in range(0, cluster_num):
#             total_cluster[n].append("")
#     #進入分群判別
#     else:
#         count = 0
#         check_list = []
#
#         if len(total_pos[i])==cluster_num:
#         #假如該行紀錄點剛好等於總分群數，則直接照順序歸類
#             for j in total_pos[i]:
#                 if j == thr_place:  # 調整基準對應值
#                     value = thr_value
#                 else:  # 將每一行的資訊做分群
#                     value = round(thr_value - (abs((thr_value - thr1_value)) / abs((thr1_place - thr_place)) *
#                                                (j - thr_place)))
#                 if clustered: # 如果歸類過，則不能直接順序排列，需要比較參考點距離
#                     dist_list = []
#                     for m in range(0,cluster_num):
#                         try:
#                             dist = abs(pre_locate[m] - j)
#                             dist_list.append(dist)
#                         except:
#                             dist_list.append([])
#                     place = dist_list.index(min(a for a in dist_list if isinstance))
#                     if min(a for a in dist_list if isinstance) > 10:
#                         if cluster_count >= cluster_num:
#                             pass
#                         else:
#                             try:
#                                 if check_list.index(place):
#                                     print("重複", gg1,"在",i,"的",j)
#                                     gg1 = gg1 + 1
#                             except:
#                                 print("新分群", gg, "在", i, "的", j)
#                                 gg = gg + 1
#                                 total_cluster[cluster_count].append(value)
#                                 pre_locate[cluster_count] = j
#                                 check_list.append(cluster_count)
#                                 cluster_count = cluster_count + 1
#                         # 距離小則直接分類
#                     else:
#                       #  place = dist_list.index(min(a for a in dist_list if isinstance))
#                         try:
#                             if check_list.index(place):
#                                 print("重複", gg1, "在", i, "的", j)
#                                 gg1 = gg1 + 1
#                                 if cluster_count >= cluster_num:
#                                     pass
#                                 else:
#                                     print("新分群",gg,"在",i,"的",j)
#                                     total_cluster[cluster_count].append(value)
#                                     pre_locate[cluster_count]=j
#                                     check_list.append(cluster_count)
#                                     cluster_count=cluster_count+1
#                                     gg=gg+1
#                         except:
#                             total_cluster[place].append(value)
#                             pre_locate[place] = j
#                             check_list.append(place)
#                 #如果未分群過，則直接分群，並定義參考點
#                 else:
#                     try:
#                         if check_list.index(place):
#                             print("重複", gg1, "在", i, "的", j,"第",place)
#                             gg1 = gg1 + 1
#                     except:
#                         total_cluster[count].append(value)
#                         pre_locate[count]=j
#                         count=count+1
#         elif len(total_pos[i])>cluster_num:
#         # 假如該行紀錄點大於總分群數，則忽略排序
#             for n in range(0, cluster_num):
#                 total_cluster[n].append("")
#         else:
#             clustered=True
#             #假如否，則需比較分群參考點
#             check_list=[]
#             #若某一分群無紀錄點，則將該列的分群補上空集合
#             for j in total_pos[i]:
#                 if j == thr_place:  # 調整基準對應值
#                     value = thr_value
#                 else:  # 將每一行的資訊做分群
#                     value  = round(thr_value - (abs(((thr_value-thr1_value))) / abs((thr1_place-thr_place)) * (j - thr_place)))
#                    # value = round(120 - (110 / 384) * (j - thr_place))
#                 dist_list = []
#                 #dist_list紀錄該行每一記錄點與分群參考點的差值
#                 for k in range(0,cluster_num):
#                     try:
#                         dist=abs(pre_locate[k]-j)
#                         dist_list.append(dist)
#                     except:
#                         dist_list.append([])
#                 try:
#                     #假如距離太大則歸類新分類
#                     if min(a for a in dist_list if isinstance)>10:
#                         if cluster_count>=cluster_num:
#                             pass
#
#                         else:
#                             print("新分群", gg, "在", i, "的", j)
#                             total_cluster[cluster_count].append(value)
#                             pre_locate[cluster_count]=j
#                             check_list.append(cluster_count)
#                             cluster_count=cluster_count+1
#                             gg = gg + 1
#                     #距離小則直接分類
#                     else:
#                         place = dist_list.index(min(a for a in dist_list if isinstance))
#                         try:#如果有重複的分類則須再討論
#                             if check_list.index(place):
#                                 print("重複", gg1, "在", i, "的", j,"第",place)
#                                 gg1=gg1+1
#                                 if cluster_count>=cluster_num:
#                                     pass
#                                 else:
#                                     print("新分群",gg,"在",i,"的",j)
#                                     total_cluster[cluster_count].append(value)
#                                     pre_locate[cluster_count]=j
#                                     check_list.append(cluster_count)
#                                     cluster_count=cluster_count+1
#                                     gg=gg+1
#                         except:
#                             total_cluster[place].append(value)
#                             pre_locate[place] = j
#                             check_list.append(place)
#                 except:
#                     #假如抱錯 則表示完全沒有參考點，直接定義第一個參考點
#                     #place = dist_list.index(min(a for a in dist_list if isinstance))
#                     print("kk")
#                     total_cluster[0].append(value)
#                     pre_locate[0] = j
#                     check_list.append(0)
#                     cluster_count = cluster_count + 1
#             '''    else:
#                     place = dist_list.index(min(a for a in dist_list if isinstance))
#                     total_cluster[place].append(value)
#                     pre_locate[place]=j
#                     check_list.append(place)
#                     cluster_count=cluster_count+1'''
#             for l in range(0,cluster_num):
#                 try:
#                     if check_list.index(l):
#                         pass
#                 except:
#                     total_cluster[l].append("")
#         '''
#         for j in total_point[i]:
#             if j == 62:  # 調整基準對應值
#                 value = 120
#             else:  # 將每一行的資訊做分群
#                 value = round(120 - (110 / 384) * (j - 62))
#             if j < 120:
#                 total_cluster[0].append(value)
#                 check_list.append(0)
#             elif j > 120 and j < 180:
#                 total_cluster[1].append(value)
#                 check_list.append(1)
#             elif j > 180 and j < 260:
#                 total_cluster[2].append(value)
#                 check_list.append(2)
#             elif j > 260 and j < 390:
#                 total_cluster[3].append(value)
#                 check_list.append(3)
#             else:
#                 total_cluster[4].append(value)
#                 check_list.append(4)
#
#         check = 0
#         # 若該行無該Cluster的資訊，則補上空集合
#         for i in range(0, cluster_num):
#             try:
#                 if check_list.index(i):
#                     pass
#             except:
#                 total_cluster[i].append("")
#         check_list = []
# '''
#         '''3/21未完成
#             find_cluster = []
#             for k in range(0,cluster_num):
#                 try:
#                     #與各total_cluster比較大小、相似度，若沒有符合，則跳至創建新的分群
#                     a=abs(j-pre_locate[k][0])#取每一total_cluster座標點最後值比較
#                     find_cluster.append(a)
#
#                 except:
#                     #如果沒有比較樣本，或比較樣本低於總分群數則跳出
#                     pass
#
#             try:
#                 for l in find_cluster:#確認分群的規則
#                     #如果與該分群最後座標點差異甚小 則確定座標點分類為該分群 且紀錄分群最後座標
#                     if min(find_cluster)<10:
#                         a = find_cluster.index(min(find_cluster))
#                         total_cluster[a].append(value)
#                         pre_locate[a][0]=j
#                     #反之若差異甚大 則歸類座標點為新分群
#                     else:
#                         total_cluster[count].append(value)
#                         pre_locate[count][0]=j
#                         count=count+1
#             except:
#                 #倘若並無Find_cluster的值在裏頭，則確立新的分群
#                 total_cluster[count].append(value)
#                 pre_locate[count][0]=j
#                 count=count+1
#                 '''
# #book = xlwt.Workbook()
# sheet2 = book.add_sheet('sheet2')
#
# for k in range(0,cluster_num):
#     for i,e in enumerate(total_cluster[k]):
#         sheet2.write(i,k,str(total_cluster[k][i]))
#
# name = "new_data.xls"
# book.save(name)
# book.save(TemporaryFile())
#
# '''
# for g in range(0,b):#每一行
#     k = np.empty((1, 3))
#     k1 = np.empty((1, 3))
#     for h in range(0,a):#每一列
#         if h ==0:
#             k[0]=img[0,g,:]
#         else:
#             k1[0] = img[h, g, :]
#             k = np.append(k,k1,axis=0)
#     k = k.astype(int)
#
#     lists=np.zeros(a)
#     cluster = 0
#     start = 0
#     [a,b]=np.shape(k)
#     dist_count=[]
#     for i in range(0,a):
#         if i == 0:
#             pre = k[0]
#
#         else: #ex:i=2
#             now= k[i]
#             temp=i-1
#             h_dist=min(abs(pre[0]-now[0]),180-abs(pre[0]-now[0]))
#             s_dist=abs(pre[1]-now[1])
#             v_dist=abs(pre[2]-now[2])
#             if v_dist<20 and s_dist<20:
#                 dist=0
#                 dist_count.append(dist)
#         #    elif h_dist<10 and s_dist<10:
#          #       dist=round(0.6*np.sqrt(h_dist*h_dist+s_dist*s_dist+v_dist*v_dist))
#          #       dist_count.append(dist)
#             else:
#                 dist=round(np.sqrt(h_dist*h_dist+s_dist*s_dist+v_dist*v_dist))
#                 dist_count.append(dist)
#
#                         dist_list=[]
#                         count=0
#
#                         for i in range(0,len(dist_count)):
#                             if dist_count[i]<10:
#                                 count=count+1
#                             else:
#                                 print(dist_count[i])
#
#                         for i in range(0,len(dist_count)):
#                             if dist_count[i]>40:
#                                 a=dist_count[i+1]-dist_count[i]
#                                 print(a)
#                                 dist_list.append([i,a])
#                         a, b = np.shape(dist_list)
#                         for i in range(0, a):
#                             if i == 0:
#                                 pre1 = dist_list[i][0]
#                             else:
#                                 now1 = dist_list[i][0]
#                                 if pre1 - now1 == -1:
#                                     pre1 = now1
#
#                                 elif pre1 - now1 == -2:
#                                     print(now1 - 1)
#                                 else:
#                                     # print(dist_list[i][0])
#                                     pre1 = now1
#
#             if dist > 50:   #pre=list[1] now=list[2] #ex i =254
#                 if start == temp:
#                     lists[start]=cluster
#                 else:
#                     for j in range(start,i):
#                         lists[j] = cluster
#                 start = i
#                 pre = now
#                 cluster = cluster+1
#             elif i == len(lists)-1: #END cluster
#                 for j in range(start,len(lists)):
#                     lists[j]=cluster
#             else:
#                 pre = now
#     type_ = 0
#     point = []
#     for i in range(0, np.int(max(lists))+1):
#         long = np.size(np.where(lists == i))
#         if long < 10: #分群分布低的 才會是線的分群
#             range_ = np.where(lists == i)
#             mid=round((long+1)/2)-1
#             point.append(range_[0][mid])
#
#     for i in range(0,np.size(point)):#double check 將漸層顏色進一步討論 避免歸類不同類別
#         if i+1<=np.size(point):
#             if point[i+1]-point[i]<5: #i=0 point=66,i=1 point=68 ,i=2 point=150
#
#                 img[point[i+1],g,:]
#                 img[point[i],g,:]
#
#         else:
#             continue
#
#     total_point.append(point)
#
# #整理所有的節點
# for i in range(0,np.size(total_point)):
#     if total_point[i]==[]:
#         total_point[i]='None'
#
#
#
# book = xlwt.Workbook()
# sheet1 = book.add_sheet('sheet1')
#
# for i,e in enumerate(total_point):
#     sheet1.write(i,0,str(total_point[i]))
#
# name = "random2.xls"
# book.save(name)
# book.save(TemporaryFile())
#
#
# print("Finsh")
# '''

img = cv2.imread("C:/Users/Burny/PycharmProjects/remake/venv/4.jpg")
[a, b, c] = np.shape(img)  # a=484 b=996,c=3

rows, cols, channels = img.shape
mask = np.zeros([rows, cols, 3], dtype=np.uint8)
origin = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

image,contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    # if cv2.contourArea(cnt) > 10 :
    #
    #     cv2.drawContours(img, cnt, -1, (0, 0, 255), 3)
    cv2.drawContours(img, cnt, -1, (0, 0, 255), 3)

cv2.imshow("img", img)

cv2.waitKey()
