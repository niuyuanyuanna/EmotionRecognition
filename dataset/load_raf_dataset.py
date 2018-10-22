#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/8 16:02
# @Author  : NYY
# @Site    : www.niuyuanyuanna.git.io
# @File    : load_raf_dataset.py
# 用于生成train和test的txt格式索引list
import os

from config.configs import config


def load_label_list(label_path):
    image_list = []
    label_list = []
    with open(label_path, 'rb') as fid:
        for line in fid:
            line = line.decode()
            line = line.strip().split(' ')
            origin_image_name = line[0]
            split_arr = origin_image_name.split('.')
            aligned_image_name = split_arr[0] + '_aligned.jpg'
            image_list.append(aligned_image_name)
            label_list.append(line[-1])
    return image_list, label_list


def creat_dataset_txt(file_name, image_list, label_list):
    saved_file = os.path.join(config.dataset.raf.label_list_path, file_name)
    with open(saved_file, 'w+') as f:
        for i in range(len(image_list)):
            f.write(image_list[i] + '\t' + label_list[i] + '\n')


def creat_total_txt():
    origin_label_path = os.path.join(config.dataset.raf.label_list_path, 'list_patition_label.txt')
    image_list, label_list = load_label_list(origin_label_path)
    creat_dataset_txt('total_image_label.txt', image_list, label_list)


def split_dataset_list(label_path):
    train_image_list = []
    train_label_list = []
    test_image_list = []
    test_label_list = []
    with open(label_path, 'rb') as fid:
        for line in fid:
            line = line.decode()
            line = line.strip().split('\t')
            image_name = line[0]
            image_usage = image_name.split('_')[0]
            if image_usage == 'train':
                train_image_list.append(image_name)
                train_label_list.append(line[-1])
            else:
                test_image_list.append(image_name)
                test_label_list.append(line[-1])
    return train_image_list, train_label_list, test_image_list, test_label_list


def creat_split_txt():
    saved_label_path = os.path.join(config.dataset.raf.label_list_path, 'total_image_label.txt')
    train_image_list, train_label_list, \
    test_image_list, test_label_list = split_dataset_list(saved_label_path)
    creat_dataset_txt('train_image_label.txt', train_image_list, train_label_list)
    creat_dataset_txt('test_image_label.txt', test_image_list, test_label_list)


def load_normal_list(label_path):
    image_list = []
    label_list = []
    with open(label_path, 'rb') as fid:
        for line in fid:
            line = line.decode()
            line = line.strip().split('\t')
            image_name = line[0]
            image_list.append(image_name)
            label_list.append(line[-1])
    return image_list, label_list


if __name__ == '__main__':
    creat_total_txt()
    creat_split_txt()