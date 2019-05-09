#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/8 21:26
# @Author  : NYY
# @Site    : niuyuanyuanna.github.io
# @File    : recover_images.py
import os
from shutil import copyfile


def reformate_images(file_path, images_path, new_images_path, erro_file):
    with open(file_path, 'r') as fid:
        for index, line in enumerate(fid):
            raw = line.split('\t')
            image_ori_path = raw[0]
            image_path_raw = image_ori_path.split('/')
            image_name = image_path_raw[-2] + '/' + image_path_raw[-1]
            image_source_path = images_path + image_name
            image_target_path = new_images_path + image_name
            try:
                copyfile(image_source_path, image_target_path)
            except IOError as e:
                print("Unable to copy file:{}. erro:{}".format(image_name, e))
                with open(erro_file, 'a') as fe:
                    fe.write(line)


def make_sure_file_exists(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


if __name__ == '__main__':
    val_txt_file = '/home/nyy/mnt/val_cl.txt'
    train_txt_file = '/home/nyy/mnt/train_cl.txt'
    erro_images = '/home/nyy/mnt/erro_images.txt'
    images_path = '/home/nyy/mnt/DataSet/Manually_Annotated_Images/'
    new_val_images_path = '/home/nyy/dataset/AffectNet/validation/'
    new_train_images_path = '/home/nyy/dataset/AffectNet/train/'

    make_sure_file_exists(new_train_images_path)
    make_sure_file_exists(new_val_images_path)
    reformate_images(val_txt_file, images_path, new_val_images_path, erro_images)
    "/home/nyy/mnt/Dataset/Manually_Annotated_Images/"


