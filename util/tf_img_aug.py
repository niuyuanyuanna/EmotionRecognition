#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/24 15:07
# @Author  : NYY
# @Site    : www.niuyuanyuanna.git.io
# @File    : tf_img_aug.py
import tensorflow as tf
from scipy.misc import imrotate
import numpy as np


def do_random_crop(image_data, size):
    return tf.image.resize_image_with_crop_or_pad(image_data, size, size)


def do_random_rotation(image_data, angle):
    random_angle = np.random.uniform(low=-angle, high=angle)
    image_data = imrotate(image_data, random_angle, 'bicubic')
    return image_data


def grascal(image_data):
    return tf.image.rgb_to_grayscale(image_data)


def saturation(image_data):
    return tf.image.random_saturation(image_data, lower=0.5, upper=0.5)


def brightness(image_data):
    return tf.image.random_brightness(image_data, max_delta=0.5)


def contrast(image_data):
    return tf.image.random_contrast(image_data, lower=0.5, upper=0.5)


def hue(image_data):
    return tf.image.random_hue(image_data, max_delta=0.2)


def horizontal_flip(image_data):
    return tf.image.random_flip_left_right(image_data)


def vertical_flip(image_data):
    return tf.image.flip_up_down(image_data)


def normalize(image_data):
    return tf.image.per_image_standardization(image_data)


def transform(image_data, aug_strategy):
    if aug_strategy.random_crop:
        image_data = do_random_crop(image_data)
    if aug_strategy.random_rotate:
        image_data = do_random_rotation(image_data)
    if aug_strategy.random_brightness:
        image_data = brightness(image_data)
    if aug_strategy.random_saturation:
        image_data = saturation(image_data)
    if aug_strategy.random_contrast:
        image_data = contrast(image_data)
    if aug_strategy.random_lighting:
        image_data = hue(image_data)
    if aug_strategy.random_lf_flip:
        image_data = horizontal_flip(image_data)
    if aug_strategy.random_updown_flip:
        image_data = vertical_flip(image_data)

    if aug_strategy.grayscal:
        image_data = grascal(image_data)
    if aug_strategy.normalize:
        image_data = normalize(image_data,)
    return image_data