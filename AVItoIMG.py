# -*- coding:utf-8 -*-

import cv2
import os

pwd = '/Users/wangzhipeng/PycharmProjects/COLLECOTR_MONITORING_SYSTEM/1.mp4'
avi_name = pwd
avi_name_1 = 'carbonBrush'
print(avi_name_1)
VideoCapture = cv2.VideoCapture(avi_name)
fps = VideoCapture.get(cv2.CAP_PROP_FPS)
size = (int(VideoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(VideoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# videoWriter = cv2.VideoWriter(avi_name, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, size)
success, frame = VideoCapture.read()
i = 0
while success:
    print(frame.shape)
    cv2.imwrite(avi_name_1 + '_' + str(i) + '.jpg', frame)
    i += 1
    success, frame = VideoCapture.read()
