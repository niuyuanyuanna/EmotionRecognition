#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/31 15:30
# @Author  : NYY
# @Site    : www.niuyuanyuanna.git.io
# @File    : preprocess_affectNet_dataset_for_DAN.py
import numpy as np
import cv2
import random
import os
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


def _load_data(imagepath, landmark, is_train, mirror_array):
    def makerotate(angle):
        rad = angle * np.pi / 180.0
        return np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]], dtype=np.float32)

    x, y = np.min(landmark, axis=0).astype(np.int32)
    w, h = np.ptp(landmark, axis=0).astype(np.int32)
    pts = (landmark - [x, y]) / [w, h]

    img = cv2.imread(imagepath.decode(), cv2.IMREAD_GRAYSCALE)
    center = [0.5, 0.5]

    if is_train:
        pts = pts - center
        pts = np.dot(pts, makerotate(np.random.normal(0, 20)))
        pts = pts * np.random.normal(0.8, 0.05)
        pts = pts + [np.random.normal(0, 0.05),
                     np.random.normal(0, 0.05)] + center

        pts = pts * FLAGS.img_size

        R, T = getAffine(landmark, pts)
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

        R, T = getAffine(landmark, pts)
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
