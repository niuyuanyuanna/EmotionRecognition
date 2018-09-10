#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/27 12:20
# @Author  : NYY
# @File    : leNet_inference.py
# @Software: PyCharm
import tensorflow as tf

# 配置神经网络的参数
INPUT_NODE = 2304
OUTPUT_NODE = 7

IMAGE_SIZE = 48
NUM_CHANNELS = 1
NUM_LABELS = 7

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 64
CONV1_SIZE = 3

# 第二层卷积层的尺寸和深度
CONV2_DEEP = 128
CONV2_SIZE = 3

# 第三层卷积层的尺寸和深度
CONV3_DEEP = 256
CONV3_SIZE = 3
# 第四层卷积层的尺寸和深度
CONV4_DEEP = 256
CONV4_SIZE = 3

# 第五层卷积层的尺寸和深度
CONV5_DEEP = 512
CONV5_SIZE = 3

# 第六层卷积层的尺寸和深度
CONV6_DEEP = 512
CONV6_SIZE = 3

# 第七层卷积层的尺寸和深度
CONV7_DEEP = 512
CONV7_SIZE = 3

# 第八层卷积层的尺寸和深度
CONV8_DEEP = 512
CONV8_SIZE = 3

# 全连接层1的尺寸
FUL1_SIZE = 2048

# 全连接层2的尺寸
FUL2_SIZE = 1024


# 定义卷积神经网络的前向传播过程。这里添加一个新的参数train，用于区分训练过程和测试过程。在这个程序中将用到Ddropout方法，dropout可以进一步提升模型可靠性
# 并防止过拟合，dropout过程只在训练时使用。
def inference(input_tensor, train, regularizer):

    # 第一层卷积层
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))

        # 使用边长为5，深度为32的过滤器，过滤器移动的步长为1，且使用全0填充。
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # 池化层
    with tf.name_scope('layer1-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # 第二层卷积层
    with tf.variable_scope('layer2-conv2'):
        conv2_weights = tf.get_variable("weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))

        # 使用边长为4，深度为32的过滤器，过滤器移动的步长为1，使用全0填充
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # 池化层
    with tf.name_scope('layer2-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # 第三层卷积层
    with tf.variable_scope('layer3-conv3'):
        conv3_weights = tf.get_variable("weights", [CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [CONV3_DEEP], initializer=tf.constant_initializer(0.0))
        # 使用边长为5，深度为120的过滤器，过滤器移动的步长为1，不使用全0填充
        conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))

    # 第四层卷积层
    with tf.variable_scope('layer4-conv4'):
        conv4_weights = tf.get_variable("weights", [CONV4_SIZE, CONV4_SIZE, CONV3_DEEP, CONV4_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable("bias", [CONV4_DEEP], initializer=tf.constant_initializer(0.0))
        # 使用边长为5，深度为120的过滤器，过滤器移动的步长为1，不使用全0填充
        conv4 = tf.nn.conv2d(relu3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))

    # 池化层
    with tf.name_scope('layer3-pool3'):
        pool3 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # 第五层卷积层
    with tf.variable_scope('layer5-conv5'):
        conv5_weights = tf.get_variable("weights", [CONV5_SIZE, CONV5_SIZE, CONV4_DEEP, CONV5_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv5_biases = tf.get_variable("bias", [CONV5_DEEP], initializer=tf.constant_initializer(0.0))
        # 使用边长为5，深度为120的过滤器，过滤器移动的步长为1，不使用全0填充
        conv5 = tf.nn.conv2d(pool3, conv5_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu5 = tf.nn.relu(tf.nn.bias_add(conv5, conv5_biases))

    # 第六层卷积层
    with tf.variable_scope('layer6-conv6'):
        conv6_weights = tf.get_variable("weights", [CONV6_SIZE, CONV6_SIZE, CONV5_DEEP, CONV6_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv6_biases = tf.get_variable("bias", [CONV6_DEEP], initializer=tf.constant_initializer(0.0))
        # 使用边长为5，深度为120的过滤器，过滤器移动的步长为1，不使用全0填充
        conv6 = tf.nn.conv2d(relu5, conv6_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu6 = tf.nn.relu(tf.nn.bias_add(conv6, conv6_biases))

    # 池化层，输出6x6x512
    with tf.name_scope('layer4-pool4'):
        pool4 = tf.nn.max_pool(relu6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # 将第六层卷积层的输出转化为第七层全连接层的输入格式。第六层的输出为5x5x64的矩阵，然而第六层全连接层需要的输入格式为向量，
    # 所以在这里需要将这个5x5x64的矩阵拉直成一个向量。pool3.get_shape函数可以得到第六层输出矩阵的维度而不需要手工计算。注意因为
    # 每一层神经网络的输入输出都为一个batch的矩阵，所以这里得到的维度也包含了一个batch中数据的个数。
    pool_shape = pool4.get_shape().as_list()

    # 计算将矩阵拉直成向量之后的长度，这个长度就是矩阵长度及深度的乘积。注意这里pool_shape[0]为一个batch中数据的个数。
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    # 通过tf.reshape函数将第六层的输出变成一个batch的向量。
    reshaped = tf.reshape(pool4, [pool_shape[0], nodes])

    # 声明第七层全连接层的变量并实现前向传播过程。这一层的输入时拉直之后的一组向量，向量长度为1600，输出是一组长度为2048的向量
    # 这一层和之前在第五章中介绍的基本一致，唯一的区别就是引入了dropout的概念。dropout在训练时会随机将部分节点的输出为改为0.
    # dropout可以避免过拟合问题，从而使得模型在测试数据上的效果更好。dropout一般只在全连接层而不是卷积层或者池化层使用。
    with tf.variable_scope('layer7-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, FUL1_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 只有全连接层的权重需要加入正则化
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [FUL1_SIZE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.4)

    # 声明第八层全连接层的变量并实现前向传播过程。这一层的输入为一组长度为2048的向量，输出为一组长度为1024的向量。这一层的输出通过
    # softmax之后就得到了最后的分类结果。
    with tf.variable_scope('layer8-fc2'):
        fc2_weights = tf.get_variable("weight", [FUL1_SIZE, FUL2_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [FUL2_SIZE], initializer=tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        if train:
            fc2 = tf.nn.dropout(fc2, 0.4)

    # 声明输出层，输入为一组1024的向量，输出为一组长度为7的向量。这一层的输出通过softmax之后就得到了最后的分类结果。
    with tf.variable_scope('layer9-fc3'):
        fc3_weights = tf.get_variable("weight", [FUL2_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.get_variable("bias", [NUM_LABELS],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        logit = tf.matmul(fc2, fc3_weights) + fc3_biases

        # 返回第九层的输出
    return logit

