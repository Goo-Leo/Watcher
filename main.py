#!/usr/bin/env python
# coding=utf8

import numpy as np
import cv2

cap = cv2.VideoCapture(0) #设备号为0
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while(True):
    if cap.isOpened() == False:
        print('can not open camera')
        break
    ret, frame = cap.read() #读取图像
    if ret == False: #图像读取失败则直接进入下一次循环
        continue

	cv2.namedWindow("frame")
	cv2.imshow('frame', frame)

	mykey = cv2.waitKey(1)
	#按q退出循环，0xFF是为了排除一些功能键对q的ASCII码的影响
    if mykey & 0xFF == ord('q'):
        break

#释放资源
cap.release()
cv2.destroyAllWindows()




