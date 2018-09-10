#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/27 13:03
# @Author  : NYY
# @File    : format_batch.py
# @Software: PyCharm
import tensorflow as tf


def get_batch(file_dir):
    # 读取TFRecord文件，创建文件列表，并通过文件列表创建输入文件队列。在调用输入数据处理流程前，
    # 需要统一所有原始数据的格式并将它们存储到TFRecord文件中。
    files = tf.train.match_filenames_once(file_dir)
    filename_queue = tf.train.string_input_producer(files, shuffle=True)  # 随机打乱

    # 解析TFRecord文件里的数据
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image_raw': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                       })

    # 得到图像原始数据、标签。
    image, label = features['image_raw'], features['label']

    # 从原始图像数据解析出像素矩阵，并根据图像尺寸还原图像
    decode_image = tf.decode_raw(image, tf.uint8)
    decode_image = tf.reshape(decode_image, [28, 28, 1])
    decode_image = tf.image.per_image_standardization(decode_image)

    # 将图像和标签数据通过tf.train.shuffle_batch整理成神经网络训练时需要的batch
    min_after_dequeue = 10000
    batch_size = 3500
    capacity = min_after_dequeue + 3 * batch_size
    image_batch, label_batch = tf.train.shuffle_batch([decode_image, label],
                                                      batch_size=batch_size,
                                                      capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)

    image_batch = tf.cast(image_batch, tf.float32)

    # 返回batch数据
    return image_batch, label_batch