#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/24 14:30
# @Author  : NYY
# @Site    : www.niuyuanyuanna.git.io
# @File    : test.py
import os
import numpy as np
import cv2

from scipy.misc import imread, imresize
from dataset.load_affectNet_dataset import AffetNet_Server
from config.configs import config


def getAffine(From, To):
    FromMean = np.mean(From, axis=0)
    ToMean = np.mean(To, axis=0)

    FromCentralized = From - FromMean
    ToCentralized = To - ToMean

    FromVector = (FromCentralized).flatten()
    ToVector = (ToCentralized).flatten()

    DotResult = np.dot(FromVector, ToVector)
    NormPow2 = np.linalg.norm(FromCentralized) ** 2

    a = DotResult / NormPow2
    b = np.sum(np.cross(FromCentralized, ToCentralized)) / NormPow2

    R = np.array([[a, b], [-b, a]])
    T = ToMean - np.dot(FromMean, R)

    return R, T


def _load_data(imagepath, landmarks, is_train, mirror_array):
    def makerotate(angle):
        rad = angle * np.pi / 180.0
        return np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]], dtype=np.float32)

    srcpts = np.genfromtxt(ptspath.decode(), skip_header=3, skip_footer=1)
    x, y = np.min(srcpts, axis=0).astype(np.int32)
    w, h = np.ptp(srcpts, axis=0).astype(np.int32)
    pts = (srcpts - [x, y]) / [w, h]

    img = cv2.imread(imagepath.decode(), cv2.IMREAD_GRAYSCALE)
    center = [0.5, 0.5]

    if is_train:
        pts = pts - center
        pts = np.dot(pts, makerotate(np.random.normal(0, 20)))
        pts = pts * np.random.normal(0.8, 0.05)
        pts = pts + [np.random.normal(0, 0.05),
                     np.random.normal(0, 0.05)] + center

        pts = pts * FLAGS.img_size

        R, T = getAffine(srcpts, pts)
        M = np.zeros((2, 3), dtype=np.float32)
        M[0:2, 0:2] = R.T
        M[:, 2] = T
        img = cv2.warpAffine(img, M, (FLAGS.img_size, FLAGS.img_size))

        if any(mirror_array) and random.choice((True, False)):
            pts[:, 0] = FLAGS.img_size - 1 - pts[:, 0]
            pts = pts[mirror_array]
            img = cv2.flip(img, 1)

    else:
        pts = pts - center
        pts = pts * 0.8
        pts = pts + center

        pts = pts * FLAGS.img_size

        R, T = getAffine(srcpts, pts)
        M = np.zeros((2, 3), dtype=np.float32)
        M[0:2, 0:2] = R.T
        M[:, 2] = T
        img = cv2.warpAffine(img, M, (FLAGS.img_size, FLAGS.img_size))

    _, filename = os.path.split(imagepath.decode())
    filename, _ = os.path.splitext(filename)

    uid = str(uuid.uuid1())

    cv2.imwrite(os.path.join(FLAGS.output_dir, filename + '@' + uid + '.png'), img)
    np.savetxt(os.path.join(FLAGS.output_dir, filename + '@' + uid + '.ptv'), pts, delimiter=',')

    return img, pts.astype(np.float32)



if __name__ == '__main__':
    file_path = 'E:/liuyuan/DataCenter\AffectNet\Manually_Annotated/Manually_Annotated/Manually_Annotated_Images' \
                '/1000/6fdb358fdcd677e567da785cee6092c6bf1121ad787f94b485b9389d.jpg'
    bbox = ['32', '32', '286', '286']
    landmarks = ['114.33', '158.7', '106.05', '183.97', '104.51', '211.39', '111.4', '240.93', '118.76', '269.75',
                 '125.51', '302.85', '132.55', '333.12', '139.38', '362.09', '163.02', '374.81', '197.67', '376.97',
                 '237.41', '362.72', '278.9', '340.46', '318.35', '314.52', '346.59', '281.05', '360.07', '241.31',
                 '364.92', '199.1', '366.78', '156.23', '97.71', '125.56', '104.49', '114.75', '118.0', '115.59',
                 '132.0', '119.33', '145.88', '125.77', '175.44', '124.18', '203.88', '114.71', '235.3', '112.39',
                 '267.75', '118.4', '293.56', '134.82', '158.34', '149.15', '151.45', '170.74', '143.72', '192.7',
                 '135.9', '215.25', '129.28', '229.29', '137.61', '237.39', '149.55', '242.23', '167.22', '238.74',
                 '184.41', '233.98', '114.55', '152.87', '123.44', '143.85', '138.14', '144.86', '150.88', '152.03',
                 '137.37', '156.7', '123.62', '157.24', '211.04', '154.29', '224.73', '146.04', '242.69', '147.32',
                 '257.94', '154.41', '242.23', '158.93', '224.92', '158.11', '128.6', '275.11', '133.09', '267.78',
                 '143.8', '266.49', '154.32', '271.13', '171.77', '268.65', '199.37', '272.89', '228.09', '280.03',
                 '200.93', '303.6', '173.06', '312.76', '155.35', '312.93', '142.65', '309.95', '133.18', '299.42',
                 '133.76', '278.23', '144.1', '276.83', '155.14', '279.04', '172.34', '279.56', '220.6', '281.68',
                 '172.06', '295.85', '155.37', '294.67', '143.87', '291.55']
    emotion_label = 6

