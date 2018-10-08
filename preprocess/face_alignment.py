#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/25 19:09
# @Author  : NYY
# @Site    : www.niuyuanyuanna.git.io
# @File    : face_alignment.py
import sys
import os

import dlib
import cv2
import numpy as np

from config.configs import config
from dataset.load_fer2013_dataset import load_all_imagePath_label_list


def deal_error_image(image_file, save_txt_name):
    save_path = config.dataset.fer2013.face_image_path
    save_txt_path = os.path.join(save_path, save_txt_name)
    with open(save_txt_path, 'a') as f:
        f.write(image_file)


def face_alinement(image_file_list, config):
    detector = dlib.get_frontal_face_detector()
    cur_count = 1
    tot_count = len(image_file_list)
    for image_file in image_file_list:
        print('Processing file: {} {}/{}'.format(image_file, cur_count, tot_count))
        img = cv2.imread(image_file)
        dets, scores, idx = detector.run(img, 1, config.threshold)
        print('Number of faces detected: {}'.format(len(dets)))
        if len(dets) > 1:
            deal_error_image(image_file, 'multiface.txt')
            print('Skipping image with more then one face')
        elif len(dets) == 0:
            deal_error_image(image_file, 'noface.txt')
            print('Skipping image as no faces found')
        else:
            face = dets[0]
            (ymax, xmax, _) = img.shape
            l, t, r, b = max(face.left() - config.glow, 0), \
                         max(face.top() - config.glow, 0), \
                         min(face.right() + config.glow, xmax), \
                         min(face.left() + config.glow, ymax)
            # Proportion check
            if ((r - l) * (b - t)) / (xmax * ymax) < config.min_proportion:
                print('Image proportion too small, skipping')
            if config.window:
                win = dlib.image_window()
                win.clear_overlay()
                win.set_image(img)
                win.add_overlay(dets)
                dlib.hit_enter_to_continue()
            img = img[np.arange(t, b), :, :]
            img = img[:, np.arange(l, r), :]
            # img = cv2.resize(img, [28, 28])
            # img.save('filepath')


if __name__ == '__main__':
    image_file_list, file_label = load_all_imagePath_label_list(config.dataset.fer2013.rebuild_image_from_csv)
    face_alinement(image_file_list, config.pre_params)