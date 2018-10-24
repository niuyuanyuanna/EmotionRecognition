#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/4 11:09
# @Author  : NYY
# @Site    : www.niuyuanyuanna.git.io
# @File    : image_aug.py
import os
import cv2
import numpy as np
import scipy.ndimage as ndi
from scipy.misc import imread, imresize


class ImageGenerator(object):
    def __init__(self, train_image_names, train_image_labels, test_image_names, test_image_labels, config):
        self.config = config
        self.train_image_names = train_image_names
        self.train_image_label = train_image_labels
        self.test_image_names = test_image_names
        self.test_image_labels = test_image_labels

    def do_random_crop(self, image_array):
        height = image_array.shape[0]
        width = image_array.shape[1]
        x_offset = np.random.uniform(0, 0.3 * width)
        y_offset = np.random.uniform(0, 0.3 * height)
        offset = np.array([x_offset, y_offset])
        scale_factor = np.random.uniform(0.75, 1.25)
        crop_matrix = np.array([[scale_factor, 0],
                                [0, scale_factor]])

        image_array = np.rollaxis(image_array, axis=-1, start=0)
        image_channel = [ndi.interpolation.affine_transform(image_channel,
                         crop_matrix, offset=offset, order=0, mode='nearest',
                         cval=0.0) for image_channel in image_array]

        image_array = np.stack(image_channel, axis=0)
        image_array = np.rollaxis(image_array, 0, 3)
        return image_array

    def do_random_rotation(self, image_array):
        height = image_array.shape[0]
        width = image_array.shape[1]
        x_offset = np.random.uniform(0, 0.3 * width)
        y_offset = np.random.uniform(0, 0.3 * height)
        offset = np.array([x_offset, y_offset])
        scale_factor = np.random.uniform(0.75, 1.25)
        crop_matrix = np.array([[scale_factor, 0],
                                [0, scale_factor]])

        image_array = np.rollaxis(image_array, axis=-1, start=0)
        image_channel = [ndi.interpolation.affine_transform(image_channel,
                                                            crop_matrix, offset=offset, order=0, mode='nearest',
                                                            cval=0.0) for image_channel in image_array]

        image_array = np.stack(image_channel, axis=0)
        image_array = np.rollaxis(image_array, 0, 3)
        return image_array

    def grascal(self, image_array):
        num_image_channels = len(image_array.shape)
        if num_image_channels == 3:
            image_array = cv2.cvtColor(image_array.astype('uint8'), cv2.COLOR_RGB2GRAY).astype('float32')
        image_array = np.expand_dims(image_array, -1)
        return image_array

    def _gray_scale(self, image_array):
        return image_array.dot([0.299, 0.587, 0.114])

    def saturation(self, image_array):
        gray_scale = self._gray_scale(image_array)
        alpha = 2.0 * np.random.random() * 0.5
        alpha = alpha + 1 - 0.5
        image_array = (alpha * image_array + (1 - alpha) *
                       gray_scale[:, :, None])
        return np.clip(image_array, 0, 255)

    def brightness(self, image_array):
        alpha = 2 * np.random.random() * 0.5
        alpha = alpha + 1 - 0.5
        image_array = alpha * image_array
        return np.clip(image_array, 0, 255)

    def contrast(self, image_array):
        gray_scale = (self._gray_scale(image_array).mean() *
                      np.ones_like(image_array))
        alpha = 2 * np.random.random() * 0.5
        alpha = alpha + 1 - 0.5
        image_array = image_array * alpha + (1 - alpha) * gray_scale
        return np.clip(image_array, 0, 255)

    def lighting(self, image_array):
        covariance_matrix = np.cov(image_array.reshape(-1, 3) /
                                   255.0, rowvar=False)
        eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)
        noise = np.random.randn(3) * 0.5
        noise = eigen_vectors.dot(eigen_values * noise) * 255
        image_array = image_array + noise
        return np.clip(image_array, 0, 255)

    def horizontal_flip(self, image_array):
        if np.random.random() < 0.5:
            image_array = image_array[:, ::-1]
        return image_array

    def vertical_flip(self, image_array):
        if np.random.random() < 0.5:
            image_array = image_array[::-1]
        return image_array

    def normalize(self, image_array, r_g_b_mean, r_g_b_std, grayscal):
        if grayscal:
            image_array = image_array / 255.0
            image_array = image_array - 0.5
            image_array = image_array * 2.0
        else:
            r_g_b_mean = np.tile(
                np.array(r_g_b_mean).reshape(1, -1),
                (1, image_array.shape[0] * image_array.shape[1])).reshape(image_array.shape)
            r_g_b_std = np.tile(
                np.array(r_g_b_std).reshape(1, -1),
                (1, image_array.shape[0] * image_array.shape[1])).reshape(image_array.shape)
            image_array = np.array(image_array, dtype=np.float32)
            image_array = (image_array - r_g_b_mean) / r_g_b_std
        return image_array

    def transform(self, image_array, aug_strategy, config):
        if aug_strategy.random_crop:
            image_array = self.do_random_crop(image_array)
        if aug_strategy.random_rotate:
            image_array = self.do_random_rotation(image_array)

        if aug_strategy.random_brightness:
            image_array = self.brightness(image_array)
        if aug_strategy.random_saturation:
            image_array = self.saturation(image_array)
        if aug_strategy.random_contrast:
            image_array = self.contrast(image_array)
        if aug_strategy.random_lighting:
            image_array = self.lighting(image_array)
        if aug_strategy.random_lf_flip:
            image_array = self.horizontal_flip(image_array)
        if aug_strategy.random_updown_flip:
            image_array = self.vertical_flip(image_array)

        if aug_strategy.grayscal:
            image_array = self.grascal(image_array)
        if aug_strategy.normalize:
            image_array = self.normalize(image_array,
                                         config.dataset.raf.r_g_b_mean,
                                         config.dataset.raf.r_g_b_std,
                                         aug_strategy.grayscal)
        return image_array

    def formate_one_hot(self, label_list, num_classes=7):
        integer_classes = np.asarray(label_list, dtype='int') - 1
        num_samples = integer_classes.shape[0]
        categorical = np.zeros((num_samples, num_classes))
        categorical[np.arange(num_samples), integer_classes] = 1
        return categorical

    def flow(self, mode='train'):
        while True:
            if mode == 'train':
                image_keys = self.train_image_names
                label_list = self.train_image_label
            else:
                image_keys = self.test_image_names
                label_list = self.test_image_labels

            inputs = list()
            targets = list()
            for i, key in enumerate(image_keys):
                image_array = imread(key)
                image_array = imresize(image_array, (self.config.train.aug_strategy.resize_size,
                                                     self.config.train.aug_strategy.resize_size))
                # num_image_channels = len(image_array.shape)
                # if num_image_channels != 3:
                #     continue
                ground_truth = label_list[i]
                image_array = image_array.astype('float32')
                if mode == 'train':
                    image_array = self.transform(image_array, self.config.train.aug_strategy, self.config)
                else:
                    image_array = self.transform(image_array, self.config.test.aug_strategy, self.config)
                inputs.append(image_array)
                targets.append(ground_truth)
                if len(targets) == self.config.train.batch_size:
                    inputs = np.asarray(inputs)
                    targets = np.asarray(targets)
                    targets = self.formate_one_hot(targets)
                    yield [{'input_1': inputs}, {'predictions': targets}]
                    inputs = list()
                    targets = list()