#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/4 10:33
# @Author  : NYY
# @Site    : www.niuyuanyuanna.git.io
import os
import glob
from random import shuffle

from config.configs import config


def load_ck_data_list(file_path):
    sub_dirs = [x[0] for x in os.walk(file_path)]
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG', 'png', 'PNG']
    file_list = []
    label_list = []
    for i in range(1, len(sub_dirs) - 1):
        label_name = os.path.basename(sub_dirs[i])
        for extension in extensions:
            file_glob = os.path.join(file_path, label_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
            for one_file_path in glob.glob(file_glob):
                label_list.append(label_name)
        if not file_list:
            continue
    return file_list, label_list


def convert_label_list(label_dict, label_list):
    new_label_list = []
    for label in label_list:
        if label in label_dict:
            new_label_list.append(label_dict[label])
        else:
            new_label_list.append(-1)
    return new_label_list


def load_image_raw_list(file_list, label_list):
    temp = list(zip(file_list, label_list))
    shuffle(temp)
    file_list, label_list = zip(*temp)
    len_val = int(len(file_list) * config.train.split_val)

    val_image_path_list = file_list[: len_val]
    val_label = label_list[:len_val]

    train_image_path_list = file_list[len_val:]
    train_label = label_list[len_val:]

    return val_image_path_list, val_label, train_image_path_list, train_label
