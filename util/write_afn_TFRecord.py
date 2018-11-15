#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/15 11:32
# @Author  : NYY
# @Site    : www.niuyuanyuanna.git.io
# @File    : write_afn_TFRecord.py
# 将人脸数据集affectnet制作成TFRecords文件格式,训练前单独运行
import os
import tensorflow as tf
from PIL import Image
from random import shuffle

from config.configs import config
from dataset.load_affectNet_dataset import load_filename_list


# 生成字符型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def takeSecond(elem):
    return elem.split(sep='_')[1]


def write_TFRecord_file(input_file, output_file):
    file_name_list, landmarks_list, emotions_list = load_filename_list(input_file)
    train_dicts = list(zip(file_name_list, landmarks_list, emotions_list))
    shuffle(train_dicts)
    file_name_list, landmarks_list, emotions_list = zip(*train_dicts)
    for file_name in file_name_list:
        img = Image.open(file_name)
        size = img.size



    files = os.listdir(input_file)
    files.sort(key=takeSecond)
    label = 0
    writer = tf.python_io.TFRecordWriter(output_file)
    for file in files:
        img_path = os.path.join(input_file, file)
        img = Image.open(img_path)
        name = file.split(sep='_')
        label = name_dict[name[0]]

        # 调整图片尺寸，48x48-->28x28
        img = img.resize((resize_size, resize_size), Image.ANTIALIAS)
        # 将图像矩阵转化成一个字符串
        img_raw = img.tobytes()
        # 图像通道数
        channels = 1
        # 将一个样例转化成Example Protocol Buffer，并将所有的信息写入这个数据结构
        example = tf.train.Example(features=tf.train.Features(
            feature={'label': _int64_feature(label),
                     'image_raw': _bytes_feature(img_raw)}))

        # 将一个Example写入TFRecord文件中
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    file_dir = config.dataset.fer2013.data_path
    test_dir = config.dataset.fer2013.test_data_file
    val_dir = config.dataset.fer2013.val_data_file
    file_total_dir = config.dataset.fer2013.total_train_data_file

    # 输出TFRecord文件的地址
    filename = config.dataset.fer2013.tfRecord_path
    train_TFRecord_file_path = config.dataset.fer2013.train_TFRecord_file_path
    test_TFRecord_file_path = config.dataset.fer2013.test_TFRecord_file_path
    total_image_TFRecord_file_path = config.dataset.fer2013.total_image_TFRecord_file_path

    total_train_TFRecord_file_path = config.fer2013_total_train_TFRecord_file_path
    val_TFRecord_file_path = config.dataset.fer2013.total_train_TFRecord_file_path
    name_dict = {'anger': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprised': 5, 'normal': 6}

    write_TFRecord_file(file_total_dir, total_train_TFRecord_file_path, 28)
    write_TFRecord_file(val_dir, val_TFRecord_file_path, 28)
    write_TFRecord_file(test_dir, test_TFRecord_file_path, 28)