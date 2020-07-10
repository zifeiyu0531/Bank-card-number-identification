import cv2
import os
import numpy as np
import random


data = []
label = []


def get_img(img, file):
    blur = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯模糊
    ret, img = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)  # 二值化

    height, width = img.shape
    v = [0] * width
    z = [0] * height
    a = 0

    # 垂直投影：统计并存储每一列的黑点数
    for x in range(0, width):
        for y in range(0, height):
            if img[y, x] == 255:
                continue
            else:
                a = a + 1
        v[x] = a
        a = 0

    # 创建空白图片，绘制垂直投影图
    l = len(v)
    emptyImage = np.full((height, width), 255, dtype=np.uint8)
    for x in range(0, width):
        for y in range(0, v[x]):
            emptyImage[y, x] = 0

    # 分割字符
    Position = []
    Wstart = 0
    Wend = 0
    W_Start = 0
    W_End = 0
    v[0], v[len(v) - 1] = 0, 0
    for j in range(len(v)):
        if v[j] > 0 and Wstart == 0:
            W_Start = j
            Wstart = 1
            Wend = 0
        if v[j] <= 0 and Wstart == 1:
            W_End = j
            Wstart = 0
            Wend = 1
        if Wend == 1:
            Position.append([W_Start, 0, W_End, height])
            Wend = 0

    i = 0
    for m in range(len(Position)):
        if Position[m][3]/(Position[m][2]-Position[m][0]) > 1 and Position[m][3]/(Position[m][2]-Position[m][0]) < 5:
            temp_img = img[Position[m][1]:Position[m][3], Position[m][0]:Position[m][2]]

            temp_img = cv2.resize(temp_img, (16, 16))



            blur1 = cv2.GaussianBlur(temp_img, (1, 1), 0)  # 高斯模糊
            blur2 = cv2.GaussianBlur(temp_img, (3, 3), 0)  # 高斯模糊
            noise = sp_noise(temp_img, 0.01)

            h0, w0 = temp_img.shape
            temp_label = [0.0] * 10

            temp_data = []
            for hx in range(h0):
                for wx in range(w0):
                    temp_data.append(float(temp_img[hx, wx]))
            data.append(temp_data)

            temp_data = []
            for hx in range(h0):
                for wx in range(w0):
                    temp_data.append(float(blur1[hx, wx]))
            data.append(temp_data)

            temp_data = []
            for hx in range(h0):
                for wx in range(w0):
                    temp_data.append(float(blur2[hx, wx]))
            data.append(temp_data)

            temp_data = []
            for hx in range(h0):
                for wx in range(w0):
                    temp_data.append(float(noise[hx, wx]))
            data.append(temp_data)

            temp_data = []                                      #左移
            for hx in range(h0):
                for wx in range(w0):
                    if wx < w0-1:
                        temp_data.append(float(temp_img[hx, wx+1]))
                    else:
                        temp_data.append(255.0)
            data.append(temp_data)

            temp_data = []                                      # 右移
            for hx in range(h0):
                for wx in range(w0):
                    if wx > 0:
                        temp_data.append(float(temp_img[hx, wx - 1]))
                    else:
                        temp_data.append(255.0)
            data.append(temp_data)

            temp_data = []                                      # 上移
            for hx in range(h0):
                if hx < h0-1:
                    for wx in range(w0):
                        temp_data.append(float(temp_img[hx+1, wx]))
                else:
                    for wx in range(w0):
                        temp_data.append(255.0)
            data.append(temp_data)

            temp_data = []                                      # 下移
            for hx in range(h0):
                if hx > 0:
                    for wx in range(w0):
                        temp_data.append(float(temp_img[hx-1, wx]))
                else:
                    for wx in range(w0):
                        temp_data.append(255.0)
            data.append(temp_data)

            temp_label[int(file[i])] = 1.0
            for j in range(8):
                label.append(temp_label)
            i += 1


def sp_noise(image,prob):
    '''
    添加椒盐噪声
    prob:噪声比例
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def img_handle():
    for root, dirs, files in os.walk('./images'):
        for file in files:
            img_path = root+'/'+file
            img = cv2.imread(img_path, 0)
            get_img(img, file)
    return data, label
