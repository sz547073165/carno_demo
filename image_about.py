import cv2
import matplotlib.pyplot as plt
import numpy as np

for num in np.arange(1,7):
    img = cv2.imread('/Users/zhangmin/PycharmProjects/carno_demo/img/{}.jpg'.format(num))[100:450,0:960]
    #cv2.imshow('img',img)

    #灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #高斯滤波
    gaussian = cv2.GaussianBlur(gray, (5, 5), 0, 0, cv2.BORDER_DEFAULT)
    #cv2.imshow('gaussian',gaussian)
    #中值滤波
    median = cv2.medianBlur(gaussian, 3)
    #cv2.imshow('median',median)
    #sobel边缘检测
    sobel = cv2.Sobel(median, cv2.CV_8U, 1, 0,  ksize = 3)
    #cv2.imshow('sobel',sobel)
    #二值化
    ret, binary = cv2.threshold(sobel, 100, 255, cv2.THRESH_BINARY)
    #cv2.imshow('binary',binary)

    kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT,(33, 1))
    kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT,(27, 21))
    #膨胀图像
    dilated = cv2.dilate(binary,kernel_1)
    #显示膨胀后的图像
    #cv2.imshow("dilated",dilated);

    #腐蚀图像
    eroded = cv2.erode(dilated,kernel_2)
    #显示腐蚀后的图像
    #cv2.imshow("eroded",eroded);

    #膨胀图像，第二次
    dilated = cv2.dilate(eroded,kernel_2)
    #显示膨胀后的图像
    #cv2.imshow("dilated_{}".format(num),dilated);

    def find_carno_regin(dilated):
        region = []
        #查找轮廓
        im, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        for i in np.arange(len(contours)):
            cnt = contours[i]
            # 计算该轮廓的面积
            area = cv2.contourArea(cnt)

            # 面积小的都筛选掉
            if (area < 3000 or area > 8000):
                continue

            # 轮廓近似，作用很小
            #epsilon = 0.001 * cv2.arcLength(cnt, True)
            #approx = cv2.approxPolyDP(cnt, epsilon, True)

            # 找到最小的矩形，该矩形可能有方向
            rect = cv2.minAreaRect(cnt)
            #print("rect is: ")
            #print(rect)

            # box是四个点的坐标
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # 计算高和宽
            height = abs(box[0][1] - box[2][1])
            width = abs(box[0][0] - box[2][0])

            # 车牌正常情况下长高比在2.7-5之间
            ratio = float(width) / float(height)
            #print(ratio)
            if (ratio > 6 or ratio < 2):
                continue
            #print(area)
            region.append(box)

        return region

    region = find_carno_regin(dilated)

    img_2 = img.copy()
    def cut_carno(region):
        for i in np.arange(len(region)):
            box = region[i]
            result = cv2.drawContours(img_2, [box], -1, (0,255,0), 3)
            cv2.imshow('result_{}'.format(num),result)

            ys = [box[0, 1], box[1, 1], box[2, 1], box[3, 1]]
            xs = [box[0, 0], box[1, 0], box[2, 0], box[3, 0]]
            ys_sorted_index = np.argsort(ys)
            xs_sorted_index = np.argsort(xs)

            x1 = box[xs_sorted_index[0], 0]
            x2 = box[xs_sorted_index[3], 0]

            y1 = box[ys_sorted_index[0], 1]
            y2 = box[ys_sorted_index[3], 1]

            img_3 = img.copy()
            carno = img_3[y1:y2, x1:x2]
            cv2.imshow('carno_{}_{}'.format(num,i),carno)

    cut_carno(region)
cv2.waitKey(0)
cv2.destroyAllWindows()