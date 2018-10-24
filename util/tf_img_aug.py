#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/24 15:07
# @Author  : NYY
# @Site    : www.niuyuanyuanna.git.io
# @File    : tf_img_aug.py
import tensorflow as tf


def do_random_corp(image_data, size):
    return tf.image.resize_image_with_crop_or_pad(image_data, size, size)


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
    return  tf.image.flip_up_down(image_data)


def normalize(image_data):


