#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/5 10:22
# @Author  : NYY
# @Site    : www.niuyuanyuanna.git.io
# @File    : resnet50v2.py
import functools

import tensorflow as tf


layers = tf.keras.layers


class _IdentityBlock(tf.keras.Model):

    def __init__(self, kernel_size, filters, stage, block, data_format):
        super(_IdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        bn_axis = 1 if data_format == 'channels_first' else 3

        self.conv2a = layers.Conv2D(
            filters1, (1, 1), name=conv_name_base + '2a', data_format=data_format)
        self.bn2a = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2a')

        self.conv2b = layers.Conv2D(
            filters2,
            kernel_size,
            padding='same',
            data_format=data_format,
            name=conv_name_base + '2b')
        self.bn2b = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2b')

        self.conv2c = layers.Conv2D(
            filters3, (1, 1), name=conv_name_base + '2c', data_format=data_format)
        self.bn2c = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2c')

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)


class _ConvBlock(tf.keras.Model):

    def __init__(self, kernel_size, filters, stage, block, data_format, strides=(2, 2)):
        super(_ConvBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        bn_axis = 1 if data_format == 'channels_first' else 3

        self.conv2a = layers.Conv2D(filters1, (1, 1), strides=strides,
                                    name=conv_name_base + '2a', data_format=data_format)
        self.bn2a = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')
        self.conv2b = layers.Conv2D(filters2, kernel_size, padding='same',
                                    name=conv_name_base + '2b', data_format=data_format)
        self.bn2b = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')
        self.conv2c = layers.Conv2D(filters3, (1, 1),
                                    name=conv_name_base + '2c', data_format=data_format)
        self.bn2c = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')
        self.conv_shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                                           name=conv_name_base + '1', data_format=data_format)
        self.bn_shortcut = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '1')

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        shortcut = self.conv_shortcut(input_tensor)
        shortcut = self.bn_shortcut(shortcut, training=training)

        x += shortcut
        return tf.nn.relu(x)


class ResNet50(tf.keras.Model):

    def __init__(self, data_format, name='', trainable=True,
                 include_top=True, pooling=None, classes=7):
        super(ResNet50, self).__init__(name=name)

        valid_channel_values = ('channels_first', 'channels_last')
        if data_format not in valid_channel_values:
            raise ValueError('Unknown data_format: %s. Valid values: %s' %
                             (data_format, valid_channel_values))
        self.include_top = include_top

        def conv_block(filters, stage, block, strides=(2, 2)):
            return _ConvBlock(3, filters, stage=stage, block=block,
                              data_format=data_format, strides=strides)

        def id_block(filters, stage, block):
            return _IdentityBlock(3, filters, stage=stage, block=block, data_format=data_format)

        self.conv1 = layers.Conv2D(64, (7, 7), strides=(2, 2),
                                   data_format=data_format, padding='same', name='conv1')
        bn_axis = 1 if data_format == 'channels_first' else 3
        self.bn_conv1 = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')
        self.max_pool = layers.MaxPooling2D((3, 3), strides=(2, 2), data_format=data_format)

        self.l2a = conv_block([64, 64, 256], stage=2, block='a', strides=(1, 1))
        self.l2b = id_block([64, 64, 256], stage=2, block='b')
        self.l2c = id_block([64, 64, 256], stage=2, block='c')

        self.l3a = conv_block([128, 128, 512], stage=3, block='a')
        self.l3b = id_block([128, 128, 512], stage=3, block='b')
        self.l3c = id_block([128, 128, 512], stage=3, block='c')
        self.l3d = id_block([128, 128, 512], stage=3, block='d')

        self.l4a = conv_block([256, 256, 1024], stage=4, block='a')
        self.l4b = id_block([256, 256, 1024], stage=4, block='b')
        self.l4c = id_block([256, 256, 1024], stage=4, block='c')
        self.l4d = id_block([256, 256, 1024], stage=4, block='d')
        self.l4e = id_block([256, 256, 1024], stage=4, block='e')
        self.l4f = id_block([256, 256, 1024], stage=4, block='f')

        self.l5a = conv_block([512, 512, 2048], stage=5, block='a')
        self.l5b = id_block([512, 512, 2048], stage=5, block='b')
        self.l5c = id_block([512, 512, 2048], stage=5, block='c')

        self.avg_pool = layers.AveragePooling2D(
            (2, 2), strides=(1, 1), data_format=data_format)

        if self.include_top:
            self.flatten = layers.Flatten()
            self.fc1000 = layers.Dense(classes, name='fc1000', activation=tf.nn.softmax)
        else:
            reduction_indices = [1, 2] if data_format == 'channels_last' else [2, 3]
            reduction_indices = tf.constant(reduction_indices)
            if pooling == 'avg':
                self.global_pooling = functools.partial(
                    tf.reduce_mean,
                    reduction_indices=reduction_indices,
                    keep_dims=False)
            elif pooling == 'max':
                self.global_pooling = functools.partial(
                    tf.reduce_max, reduction_indices=reduction_indices, keep_dims=False)
            else:
                self.global_pooling = None

    def call(self, inputs, training=True):
        x = self.conv1(inputs)
        x = self.bn_conv1(x, training=training)
        x = tf.nn.relu(x)
        x = self.max_pool(x)

        x = self.l2a(x, training=training)
        x = self.l2b(x, training=training)
        x = self.l2c(x, training=training)

        x = self.l3a(x, training=training)
        x = self.l3b(x, training=training)
        x = self.l3c(x, training=training)
        x = self.l3d(x, training=training)

        x = self.l4a(x, training=training)
        x = self.l4b(x, training=training)
        x = self.l4c(x, training=training)
        x = self.l4d(x, training=training)
        x = self.l4e(x, training=training)
        x = self.l4f(x, training=training)

        x = self.l5a(x, training=training)
        x = self.l5b(x, training=training)
        x = self.l5c(x, training=training)

        x = self.avg_pool(x)

        if self.include_top:
            return self.fc1000(self.flatten(x))
        elif self.global_pooling:
            return self.global_pooling(x)
        else:
            return x
