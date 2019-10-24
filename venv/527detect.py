#detect legend
import cv2
import numpy as np
from easygui import fileopenbox

font = cv2.FONT_HERSHEY_COMPLEX
img = cv2.imread(fileopenbox())
rows,cols,channels = img.shape
mask = np.zeros([rows,cols,3],dtype=np.uint8)
img2=img.copy()
img3=img.copy()
img5=img.copy()
img6=img.copy()

img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
_, threshold = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
_, contours, _ = cv2.findContours(threshold,-1,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    if  cv2.contourArea(cnt) > 100:
        cv2.drawContours(img5, [cnt], 0, (0, 0, 255), 2)
cv2.imshow("all contours",img5)
for cnt in contours:
    if cv2.contourArea(cnt) > 100:
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt,epsilon, False)
        if len(approx) == 4:
            #確認是否為正立四邊形且不能為最外圍之邊框
            if abs(approx[0][0][0]-approx[1][0][0])<2 and abs(approx[2][0][0]-approx[3][0][0]<2 and abs(approx[0][0][1]-approx[3][0][1])<2
                                                              and abs(approx[1][0][1]-approx[2][0][1])<2 and not abs(approx[0][0][0]-approx[2][0][0])>0.8*cols):
                #確認是否有非圖例之誤偵測
                temp = threshold[approx[0][0][1]:approx[1][0][1],approx[0][0][0]:approx[2][0][0]]
                [a,b]=np.shape(temp)
                count=0
                countall=0
                for i in range(0,a):
                    for j in range(0,b):
                      if temp[i,j]==255:
                          count=count+1
                          countall=countall+1
                      else:
                          countall=countall+1
                ent=count/countall
                print(ent)
                if ent >0.95:
                    pass
                else:
                    legend=img3[approx[0][0][1]:approx[1][0][1], approx[0][0][0]:approx[2][0][0]]
                    x = approx.ravel()[0]
                    y = approx.ravel()[1]
                    cv2.putText(img2, "Legend", (x, y-20), font, 1, (0,0,255))
                    cv2.drawContours(img2, [approx], 0, (0,255,0), 2)
                    cv2.drawContours(mask, [approx], -1, (255,255,255), -1)
                    cv2.drawContours(mask, [approx], -1, (255, 255, 255),7 )


img4=img3.copy()


img4=cv2.add(img3,mask)
cv2.imshow("legend",legend)
cv2.imwrite("legend.jpg",legend)
cv2.imshow("ex",img2)
cv2.imshow("mask", mask)

#cv2.imshow("after detect", img4)
x=30
y=50
cv2.putText(img6,"Origin",(x,y), font, 1, (0,0,255),3)
cv2.putText(img4,"RemoveLegend",(x,y), font, 1, (0,0,255),3)
final=cv2.hconcat([img6,img4])
cv2.imshow("final",final)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("OutLegend.jpg",img4)