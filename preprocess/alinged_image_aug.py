#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/8 17:23
# @Author  : NYY
# @Site    : www.niuyuanyuanna.git.io
# @File    : alinged_image_aug.py
import cv2
import math
import random
from PIL import ImageEnhance, Image
import numpy as np


# horizontally
def flip_image(src_image):
    return cv2.flip(src_image, 1)


def rotate_image(src_image, angle):
    width = src_image.shape[1]
    height = src_image.shape[0]
    radian = angle/180.0*math.pi
    sin = math.sin(radian)
    cos = math.cos(radian)
    new_width = int(abs(width * cos) + abs(height * sin))
    new_height = int(abs(width * sin) + abs(height * cos))
    """
    [ a   b   (1-a)*center_x-b*center_y,
      -b  a   b*center_x - (1-a)*center_y
    ]
    a = scale*cos
    b = scale*sin
    """''
    rotate_matrix = cv2.getRotationMatrix2D((width/2.0, height/2), angle, 1.0)
    rotate_matrix[0, 2] += (new_width - width) / 2.0
    rotate_matrix[1, 2] += (new_height - height) / 2.0
    dst_img = cv2.warpAffine(src_image, rotate_matrix, (new_width, new_height), flags=cv2.INTER_LINEAR)
    return dst_img


def random_crop(src_image, max_jitter=5, keep_size=True):
    width = src_image.shape[1]
    height = src_image.shape[0]
    roi = [math.floor(random.uniform(0, max_jitter)), math.floor(random.uniform(0, max_jitter)),
           math.ceil(width-random.uniform(0, max_jitter)), math.ceil(height-random.uniform(0, max_jitter))]
    img_roi = src_image[roi[1]:roi[3]+1, roi[0]:roi[2]+1, :]
    if keep_size:
        img_roi = cv2.resize(img_roi, (width, height), interpolation=cv2.INTER_LINEAR)
    return img_roi


def random_color(src_image):
    """
    :param src_image:
    :return: image after random color, brightness, contrast, sharpness adjustment
    """
    src_image = Image.fromarray(src_image)
    random_factor = np.random.randint(0, 31) / 10.
    color_image = ImageEnhance.Color(src_image).enhance(random_factor)
    random_factor = np.random.randint(10, 21) / 10.
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)
    random_factor = np.random.randint(10, 21) / 10.
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)
    random_factor = np.random.randint(0, 31) / 10.
    return np.array(ImageEnhance.Sharpness(contrast_image).enhance(random_factor))


def normalize(image, b_g_r_mean, b_g_r_std):
    b_g_r_mean = np.tile(
        np.array(b_g_r_mean).reshape(1, -1),
        (1, image.shape[0] * image.shape[1])).reshape(image.shape)
    b_g_r_std = np.tile(
        np.array(b_g_r_std).reshape(1, -1),
        (1, image.shape[0] * image.shape[1])).reshape(image.shape)
    image = np.array(image, dtype=np.float32)
    image = (image - b_g_r_mean) / b_g_r_std
    return image


def aug_img_func(image, aug_strategy, config):
    if aug_strategy.flip:
        seed = random.random()
        if seed > 0.7:
            image = flip_image(image)
    if aug_strategy.random_rotate:
        max_angle = aug_strategy.max_rotate_angle
        rotate_angle = random.randint(-max_angle, max_angle)
        image = rotate_image(image, rotate_angle)
    if aug_strategy.random_crop:
        image = random_crop(image)
    if aug_strategy.random_color:
        image = random_color(image)
    if aug_strategy.normalize:
        image = normalize(image, config.dataset.b_g_r_mean, config.dataset.b_g_r_std)
    return image