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
    file_path = '/home/liuyuan/DataCenter/AffectNet/Manually_Annotated/Images/521/' \
                'test.jpg'
    bbox = [191, 191, 1276, 1276]
    landmarks = ['44.47', '120.93', '45.74', '141.13', '46.89', '160.62', '49.89', '182.16', '57.17', '201.0',
                 '70.26', '217.13', '87.84', '230.46', '107.79', '240.33', '128.89', '241.32', '149.87', '238.89',
                 '169.51', '229.33', '186.33', '215.36', '197.88', '198.23', '204.17', '178.87', '207.83', '159.43',
                 '209.82', '139.9', '211.79', '120.7', '56.37', '95.14', '66.3', '81.96', '81.11', '76.78',
                 '96.83', '76.28', '110.78', '80.62', '138.55', '77.13', '153.4', '74.22', '169.04', '74.04',
                 '183.75', '79.79', '191.57', '92.04', '124.05', '96.71', '123.45', '108.4', '123.07', '119.89',
                 '122.46', '131.64', '105.98', '146.2', '114.45', '148.15', '123.08', '149.66', '132.19', '147.67',
                 '140.83', '146.5', '73.87', '107.0', '82.01', '101.39', '91.5', '100.81', '99.82', '106.24',
                 '91.74', '107.76', '82.1', '107.84', '147.99', '105.36', '157.5', '100.53', '167.93', '101.58',
                 '176.38', '107.15', '167.99', '107.74', '157.75', '107.21', '96.48', '181.59', '105.74', '168.97',
                 '116.16', '163.54', '124.25', '165.13', '132.32', '163.58', '145.34', '169.07', '155.0', '181.59',
                 '145.64', '191.98', '133.3', '195.2', '124.21', '195.9', '115.9', '194.86', '105.88', '190.74',
                 '101.29', '180.54', '116.25', '170.84', '124.25', '170.97', '132.49', '170.44', '149.97', '180.71',
                 '132.93', '186.68', '124.45', '187.45', '116.38', '186.46']

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

