import cv2
import numpy as np

def image_process():
    img = cv2.imread('test_images/4.jpg', 0)
    blur = cv2.GaussianBlur(img, (5, 5), 0)     #高斯模糊

    #dstx = cv2.Sobel(blur, -1, 1, 0, ksize=5)   #边缘检测
    #dsty = cv2.Sobel(blur, -1, 0, 1, ksize=5)
    #lap = cv2.Laplacian(blur, -1)

    #ret, binary = cv2.threshold(lap, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    #binary = cv2.adaptiveThreshold(lap, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)
    ret, binary = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)       #二值化

    kernel = np.ones((1, 70), np.uint8)
    erosion = cv2.erode(binary, kernel)         # 膨胀
    dilation = cv2.dilate(erosion, kernel)      # 腐蚀

    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sp = dilation.shape
    x, y, w, h = 0, 0, 0, 0
    for i in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        if h > sp[0]*0.05 and w > sp[1]*0.5 and y > sp[0]*0.2 and y < sp[0]*0.8 and w/h > 5:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 1)
            break

    cv2.imshow('jpeg', img)
    img = binary[y:y+h, x:x+w]

    cv2.imshow('jpg', img)
    cv2.waitKey(0)


#def num_split():
