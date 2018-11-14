#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/12 14:49
# @Author  : NYY
# @Site    : www.niuyuanyuanna.git.io
# @File    : train_afn_dan.py
import os
import tensorflow as tf

from dataset.load_affectNet_dataset import load_filename_list
from config.configs import config


def _parse_function(filename, landmarks, emotion_label):
    image_string = tf.read_file(filename)
    image_decode = tf.image.decode_image(image_string)
    image_resized = tf.image.resize_images(image_decode, [224, 224])



def get_image_data():
    train_csv = os.path.join(config.dataset.afn.csv_data, 'train_filted.csv')
    file_name, landmarks, emotions = load_filename_list(train_csv)
