#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/24 14:30
# @Author  : NYY
# @Site    : www.niuyuanyuanna.git.io
# @File    : test.py
import cv2
import numpy as np


def resize_img(img, landmarks, bbox):
    w_scale = img.shape[0] / 224.0
    h_scale = img.shape[1] / 224.0
    landmarks[:, 0] = landmarks[:, 0] / w_scale
    landmarks[:, 1] = landmarks[:, 1] / h_scale

    bbox[2] += bbox[0]
    bbox[3] += bbox[1]
    bbox = bbox / w_scale
    bbox = np.asarray(bbox, int)

    img = cv2.resize(img, (224, 224))
    return img, landmarks, bbox


def show_img(img, landmarks, bbox):
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    for i in range(len(landmarks)):
        pos = (landmarks[i][0], landmarks[i][1])
        cv2.circle(img, pos, 2, color=(0, 255, 0))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(i / 2 + 1), pos, font, 0.2, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.namedWindow("Image")
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.imwrite('E:/liuyuan/projects/EmotionRecognition/test/img1.jpg', img)
    pass


if __name__ == '__main__':
    file_path = 'E:/liuyuan/DataCenter\AffectNet\Manually_Annotated/Manually_Annotated/Manually_Annotated_Images' \
                '/1014/0a1bab1f2a4a883ce351fca47735b64e70a21d9a3d22587e64826a4f.jpg'
    bbox = [85, 85, 573, 573]
    landmarks = ['90.37', '286.24', '93.24', '359.44', '102.79', '430.09', '118.06', '499.27', '148.24', '561.17',
                 '196.37', '616.46', '251.34', '666.06', '313.28', '704.55', '381.39', '708.46', '435.64', '687.33',
                 '475.97', '637.95', '510.27', '586.7', '543.26', '527.13', '561.81', '467.02', '576.78', '409.89',
                 '586.33', '350.64', '586.55', '291.51', '150.75', '261.6', '200.23', '246.29', '246.96', '245.72',
                 '294.95', '253.18', '337.76', '271.41', '436.56', '269.95', '474.83', '253.84', '511.59', '246.15',
                 '544.08', '242.48', '573.13', '257.01', '388.01', '316.87', '390.46', '369.54', '394.3', '422.29',
                 '398.04', '475.46', '335.13', '489.49', '361.56', '500.3', '387.77', '511.34', '411.62', '500.35',
                 '430.73', '487.03', '212.51', '308.99', '243.42', '292.63', '278.76', '294.52', '307.61', '320.67',
                 '276.37', '326.18', '240.97', '324.98', '437.67', '318.87', '466.27', '294.82', '499.42', '293.83',
                 '523.01', '307.15', '503.5', '323.65', '472.42', '325.49', '260.46', '546.06', '315.79', '550.9',
                 '363.09', '550.2', '382.82', '558.51', '407.43', '548.25', '437.54', '547.94', '470.03', '542.65',
                 '438.61', '581.91', '407.14', '604.57', '381.17', '608.49', '357.0', '608.13', '313.45', '593.25',
                 '279.21', '552.49', '360.25', '569.42', '382.01', '571.94', '407.56', '565.69', '450.73', '550.56',
                 '407.03', '572.02', '380.98', '578.16', '359.46', '577.55']
    emotion_label = 0
    img = cv2.imread(file_path)
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
    for i, point in enumerate(landmarks):
        point = float(point)
        point = int(point)
        landmarks[i] = point
    landmarks = np.reshape(landmarks, (-1, 2))
    landmarks = np.asarray(landmarks)
    img = np.asarray(img)
    bbox = np.asarray(bbox)
    img, landmarks, bbox = resize_img(img, landmarks, bbox)

    show_img(img, landmarks, bbox)

