#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/20 11:28
# @Author  : NYY
# @Site    : www.niuyuanyuanna.git.io
# @File    : preprocess_fer_for_DAN.py
# 暂时放弃，因为fer2013数据集并没有标记人脸关键点
import os
import random
import numpy as np
import cv2
import uuid

import tensorflow as tf

from config.configs import config
from dataset.load_fer2013_dataset import load_all_imagePath_label_list


def getAffine(From, To):
    """
     return image array after rotate
    :param From:
    :param To:
    :return:
    """
    from_mean = np.mean(From, axis=0)
    to_mean = np.mean(To, axis=0)
    from_centralized = From - from_mean
    to_centralized = To - to_mean
    from_vector = from_centralized.flatten()
    to_vector = to_centralized.flatten()
    dot_result  = np.dot(from_vector, to_vector)
    norm_pow2 = np.linalg.norm(from_centralized) ** 2
    a = dot_result / norm_pow2
    b = np.sum(np.cross(from_centralized, to_centralized)) / norm_pow2
    R = np.array([[a, b], [-b, a]])
    T = to_mean - np.dot(from_mean, R)
    return R, T


def _load_data(imagepath, ptspath, is_train, mirror_array, image_size):
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
        pts = pts + [np.random.normal(0, 0.05), np.random.normal(0, 0.05)] + center
        pts = pts * image_size

        R, T = getAffine(srcpts, pts)
        M = np.zeros((2, 3), dtype=np.float32)
        M[0:2, 0:2] = R.T
        M[:, 2] = T
        img = cv2.warpAffine(img, M, (image_size, image_size))

        if any(mirror_array) and random.choice((True, False)):
            pts[:, 0] = image_size - 1 - pts[:, 0]
            pts = pts[mirror_array]
            img = cv2.flip(img, 1)

    else:
        pts = pts - center
        pts = pts * 0.8
        pts = pts + center
        pts = pts * image_size

        R, T = getAffine(srcpts, pts)
        M = np.zeros((2, 3), dtype=np.float32)
        M[0:2, 0:2] = R.T
        M[:, 2] = T
        img = cv2.warpAffine(img, M, (image_size, image_size))

    _, filename = os.path.split(imagepath.decode())
    filename, _ = os.path.splitext(filename)

    uid = str(uuid.uuid1())

    cv2.imwrite(os.path.join(FLAGS.output_dir, filename + '@' + uid + '.png'), img)
    np.savetxt(os.path.join(FLAGS.output_dir, filename + '@' + uid + '.ptv'), pts, delimiter=',')

    return img, pts.astype(np.float32)


def _input_fn(image_name, pts_name, is_train, mirror_array):
    dataset_image = tf.data.Dataset.from_tensor_slices(image_name)
    dataset_pts = tf.data.Dataset.from_tensor_slices(pts_name)
    dataset = tf.data.Dataset.zip((dataset_image, dataset_pts))

    dataset = dataset.prefetch(config.train.batch_size)
    dataset = dataset.repeat(1)
    dataset = dataset.map(lambda image_path, pts_path:
                          tuple(tf.py_func(_load_data,
                                           [image_path, pts_path, is_train, mirror_array],
                                           [tf.uint8, tf.float32])), num_parallel_calls=8)
    dataset = dataset.prefetch(1)
    return dataset


if __name__ == '__main__':
    imagePath_list, label_list = load_all_imagePath_label_list(config.dataset.fer2013.rebuild_image_from_csv)
    mirror_file_path = os.path.join(config.dataset.fer2013.data_path_new, 'Mirror68.txt')
    if os.path.exists(mirror_file_path):
        mirror_array = np.genfromtxt(mirror_file_path, dtype=int, delimiter=',')
    else:
        mirror_array = np.zeros(1)
    dataset = _input_fn(imagePath_list, label_list, False, mirror_array)
    next_element = dataset.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        while True:
            img, pts = sess.run(next_element)

