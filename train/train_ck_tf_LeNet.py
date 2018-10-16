#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/5 16:31
# @Author  : NYY
# @Site    : www.niuyuanyuanna.git.io
# @File    : train_ck_tf_LeNet.py
import tensorflow as tf

from config.configs import config


def pares_tf(example_proto):
    # 定义解析的字典
    dics = {}
    dics['image_raw'] = tf.FixedLenFeature(shape=[], dtype=tf.string)
    dics['label'] = tf.FixedLenFeature(shape=[], dtype=tf.int64)
    # 调用接口解析一行样本
    parsed_example = tf.parse_single_example(serialized=example_proto, features=dics)
    image = tf.decode_raw(parsed_example['image_raw'], out_type=tf.uint8)
    image = tf.reshape(image, shape=[64 * 64])
    # 有了这里的归一化处理，精度与原始数据一致
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    label = parsed_example['label']
    label = tf.cast(label, tf.int32)
    label = tf.one_hot(label, depth=7, on_value=1)
    return image, label


def load_minibatch(file_path):
    dataset = tf.data.TFRecordDataset(file_path)
    dataset = dataset.map(pares_tf)
    dataset = dataset.batch(32).repeat(1)

    iterator = dataset.make_one_shot_iterator()

    next_element = iterator.get_next()

    # 定义输入数据mnist图片大小64*64*1=2048,None表示batch_size
    x = tf.placeholder(dtype=tf.float32, shape=[None, 64 * 64], name="x")
    # 定义标签数据共7类
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, 7], name="y_")
    # 将数据调整为二维数据，w*H*c---> 64*64*1,-1表示N张
    image = tf.reshape(x, shape=[-1, 64, 64, 1])

    # 第一层，卷积核={5*5*1*32}，池化核={2*2*1,1*2*2*1}
    w1 = tf.Variable(initial_value=tf.random_normal(shape=[5, 5, 1, 32], stddev=0.1, dtype=tf.float32, name="w1"))
    b1 = tf.Variable(initial_value=tf.zeros(shape=[32]))
    conv1 = tf.nn.conv2d(input=image, filter=w1, strides=[1, 1, 1, 1], padding="SAME", name="conv1")
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, b1), name="relu1")
    pool1 = tf.nn.max_pool(value=relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    # shape={None，32,32,32}

    # 第二层，卷积核={5*5*32*64}，池化核={2*2*1,1*2*2*1}
    w2 = tf.Variable(initial_value=tf.random_normal(shape=[5, 5, 32, 64], stddev=0.1, dtype=tf.float32, name="w2"))
    b2 = tf.Variable(initial_value=tf.zeros(shape=[64]))
    conv2 = tf.nn.conv2d(input=pool1, filter=w2, strides=[1, 1, 1, 1], padding="SAME")
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, b2), name="relu2")
    pool2 = tf.nn.max_pool(value=relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool2")
    # shape={None，16,16,64}

    # 第三层，卷积核={5*5*32*64}，池化核={2*2*1,1*2*2*1}
    w3 = tf.Variable(initial_value=tf.random_normal(shape=[5, 5, 64, 128], stddev=0.1, dtype=tf.float32, name="w3"))
    b3 = tf.Variable(initial_value=tf.zeros(shape=[128]))
    conv3 = tf.nn.conv2d(input=pool2, filter=w3, strides=[1, 1, 1, 1], padding="SAME")
    relu3 = tf.nn.relu(tf.nn.bias_add(conv3, b3), name="relu3")
    pool3 = tf.nn.max_pool(value=relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool3")
    # shape={None，8,8,128}

    # FC1
    w4 = tf.Variable(initial_value=tf.random_normal(shape=[8 * 8 * 128, 1024], stddev=0.1, dtype=tf.float32, name="w3"))
    b4 = tf.Variable(initial_value=tf.zeros(shape=[1024]))
    # 关键，进行reshape
    input4 = tf.reshape(pool3, shape=[-1, 8 * 8 * 128], name="input3")
    fc1 = tf.nn.relu(tf.nn.bias_add(value=tf.matmul(input4, w4), bias=b4))
    # shape={None，1024}
    # FC2
    w5 = tf.Variable(initial_value=tf.random_normal(shape=[1024, 7], stddev=0.1, dtype=tf.float32, name="w4"))
    b5 = tf.Variable(initial_value=tf.zeros(shape=[7]))
    fc2 = tf.nn.bias_add(value=tf.matmul(fc1, w5), bias=b5)
    # shape={None，7}

    # 定义交叉熵损失
    # 使用softmax将NN计算输出值表示为概率
    y = tf.nn.softmax(fc2)

    # 定义交叉熵损失函数
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))
    # 定义solver
    train = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss=cross_entropy)

    # 定义正确值,判断二者下表index是否相等
    correct_predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # 定义如何计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_predict, dtype=tf.float32), name="accuracy")

    # 定义初始化op
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        print("start")
        sess.run(fetches=init)
        i = 0
        try:
            while True:
                # 通过session每次从数据集中取值
                image, label = sess.run(fetches=next_element)
                sess.run(fetches=train, feed_dict={x: image, y_: label})
                if i % 100 == 0:
                    train_accuracy = sess.run(fetches=accuracy, feed_dict={x: image, y_: label})
                    print(i, "accuracy=", train_accuracy)
                i = i + 1
        except tf.errors.OutOfRangeError:
            print("end!")


if __name__ == '__main__':
    load_minibatch(config.dataset.ck.train_TFRecord_file_path)











