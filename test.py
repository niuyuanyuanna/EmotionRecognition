#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/24 14:30
# @Author  : NYY
# @Site    : www.niuyuanyuanna.git.io
# @File    : test.py
from scipy.misc import imread, imresize


if __name__ == '__main__':
    img = imread('E:\\liuyuan\\DataCenter\\DatasetCK+\\face_imgs\\anger\\anger_1.png')
    print(img.shape)