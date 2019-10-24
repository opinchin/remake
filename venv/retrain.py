import cv2
import numpy as np
import easygui

samples = np.loadtxt('generalsamples.txt')
responses = np.loadtxt('generalresponses.txt')
responses = list(responses)

str = "1"
if str == "2":
    uni_img = easygui.fileopenbox()
    im = cv2.imread(uni_img, 1)
    im3 = im.copy()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # Rgb2Gray

    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # 影像平滑，高斯平滑濾雜訊
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    ret, thresh = cv2.threshold(blur, 170, 255, cv2.THRESH_BINARY_INV)

    intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                     ord('.')]
    roismall = cv2.resize(thresh,(10, 10))
    cv2.imwrite("dot.jpg",roismall)
    cv2.imshow('norm', im)
    key = cv2.waitKey(0)
    if key == 27:  # (escape to quit)
        sys.exit()
    elif key in intValidChars:
        responses.append(key)
        sample = roismall.reshape((1, 100))
        samples = np.append(samples, sample, 0)
    print("training complete")
    np.savetxt('generalsamples.txt', samples)
    np.savetxt('generalresponses.txt', responses)
    print('Finsh')

while str == "0" or str == "1":

    uni_img = easygui.fileopenbox()
    im = cv2.imread(uni_img, 1)
    im3 = im.copy()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # Rgb2Gray
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # 影像平滑，高斯平滑濾雜訊
 #   thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    ret, thresh = cv2.threshold(blur, 170, 255, cv2.THRESH_BINARY_INV)
    # Now finding Contours
    cv2.imshow("",thresh)
    cv2.waitKey()
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 找邊緣，改過第二參數。contours為list,hierarchy表各輪廓屬性，本程式沒有運用。

    if str == "0":
        samples = np.empty((0, 100), np.float32)
        responses = []
    else:
        pass
    intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                     ord('.')]
    keys = [i for i in range(48, 58)]

    for cnt in contours:

        if cv2.contourArea(cnt) > 10:   # 根據輪廓大小分配不同運算
            [x, y, w, h] = cv2.boundingRect(cnt)  # 替輪廓加上邊框是用來標示圖片中的特定物體常用的手法
            # h為上function回傳的輪廓邊框的高。
            if h > 10:
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 畫長方形
                roi = thresh[y:y + h, x:x + w]
                roismall = cv2.resize(roi, (10, 10))
                cv2.imshow('norm', im)
                key = cv2.waitKey(0)

                if key == 27:  # (escape to quit)
                    sys.exit()

                elif key in intValidChars:
                    responses.append(key)
                    sample = roismall.reshape((1, 100))
                    samples = np.append(samples, sample, 0)
    cv2.destroyAllWindows()
    str = input("If you want to keep training press 1' \nor restart training press 0")


responses = np.array(responses, np.float32)
#responses = responses.reshape((responses.size, 1))
print("training complete")

#samples = np.float32(samples)
#responses = np.float32(responses)

np.savetxt('generalsamples.txt', samples)
np.savetxt('generalresponses.txt', responses)

print('Finsh')