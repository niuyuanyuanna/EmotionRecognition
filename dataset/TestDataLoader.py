#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/4 11:08
# @Author  : NYY
# @Site    : www.niuyuanyuanna.git.io
# @File    : TestDataLoader.py
import cv2
import logging
import numpy as np


from util.image_aug import aug_img_func


class TestDataLoader():

    def __init__(self, config, image_list, class_symbol=None):
        super(TestDataLoader, self).__init__()
        self.image_list = image_list
        self.class_symbol = class_symbol
        self.config = config
        self.input_resolution = self.config.dataset.input_resolution
        self.size = len(image_list)
        logging.info('using {} for test'.format(len(self.image_list)))

    def __getitem__(self, index):
        image_path = self.image_list[index]
        image = cv2.imread(image_path, 0)
        image = aug_img_func(image, self.config.train.aug_strategy, self.config)
        image = np.asanyarray(image)
        return image

    def __len__(self):
        return self.size