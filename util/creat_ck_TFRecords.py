#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/4 14:01
# @Author  : NYY
# @Site    : www.niuyuanyuanna.git.io
# @File    : creat_ck_TFRecords.py
import tensorflow as tf
from PIL import Image

from dataset.load_ck_dataset import *


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def creat_records(image_path_list, class_symbol_list, output_record_dir):
    writer = tf.python_io.TFRecordWriter(output_record_dir)
    for i in range(len(image_path_list)):
        image = Image.open(image_path_list[i])
        image = image.resize((64, 64), Image.ANTIALIAS)
        image_bytes = image.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image_bytes),
            'label':  _int64_feature(class_symbol_list[i])
        }))
        writer.write(example.SerializeToString())
    print('write total %d image files' % len(image_path_list))
    writer.close


if __name__ == '__main__':
    if not os.path.exists(config.dataset.ck.tfRecord_path):
        os.makedirs(config.dataset.ck.tfRecord_path)
    train_record_dir = config.dataset.ck.train_TFRecord_file_path
    val_record_dir = config.dataset.ck.val_TFRecord_file_path

    file_path = config.dataset.ck.total_train_img_path
    file_list, label_list = load_ck_data_list(file_path)
    label_dict = {'anger': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neural': 4, 'sadness': 5, 'surprise': 6}
    label_id_list = convert_label_list(label_dict, label_list)

    val_image_path_list, val_label_id, train_image_path_list, \
    train_label_id, val_data_loader, train_data_loader = load_image_raw_list(file_list, label_id_list)

    creat_records(val_image_path_list, val_label_id, val_record_dir)
    creat_records(train_image_path_list, train_label_id, train_record_dir)