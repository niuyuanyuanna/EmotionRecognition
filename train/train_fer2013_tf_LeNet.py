#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/27 14:16
# @Author  : NYY
# @File    : train_fer2013_tf_LeNet.py
# @Software: PyCharm

# 定义训练过程，利用fer_2013人脸表情数据集训练网络；
# 由于fer_2013数据集中图片尺寸大小为48x48，尝试在LeNet网络和自建网络上训练模型以及测试模型。

# -*- coding: utf-8 -*-
from model.leNet_inference import inference

from dataset.format_batch import *
from config.configs import *

# 配置神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.05
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99

# 模型保存的路径和文件名。
MODEL_SAVE_PATH = config.model.tmp_tf_model_save_path
MODEL_NAME = "leNet_train_total.ckpt"

# TFRecord文件存储地址
file_dir = config.dataset.fer2013.data_path
filename = os.path.join(file_dir, 'train_total.tfrecords')


# 定义训练过程
def train():
    # 从TFRecord文件中读取图片和标签数据，组合成batch数据
    xs, ys = get_batch(filename)
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    # 直接使用mnist_interence.py中定义的前向传播过程
    y = inference(xs, True, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 定义滑动平均类和滑动平均操作
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # 定义交叉熵损失
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=ys)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 定义总损失(交叉熵损失+正则化损失)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    # 定义学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step,
                                               30000 / BATCH_SIZE, LEARNING_RATE_DECAY)
    # 定义反向传播算法更新神经网络的参数，同时更新每一个参数的滑动平均值
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')
    # 检验使用了滑动平均模型的神经网络前向传播的结果是否正确。tf.argmax(average_y,1)计算每一个样例的预测答案。
    # 其中average_y是一个batch_size*10 的二维数组，每一行表示一个样例的前向传播结果。
    # tf.argmax的第二个参数“1“表示选取最大值的操作尽在第一个维度中进行也就是说，只在每一行选取最大值对应的下标。
    # 于是得到的结果是一个长度为batch的一维数组，这个一维数组中的值就表示了每一个样例对应的数字识别结果
    # tf.equal判断两个张量的每一维是否相等，如果相等返回True，否则返回False。
    correct_prediction = tf.equal(tf.argmax(y, 1), ys)
    # 这个运算首先将一个布尔型的数字转换成实数型，然后计算平均值。这个平均值就是模型在这一组数据上的正确率。
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化Tensorflow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 变量初始化
        tf.global_variables_initializer().run()
        # 由于train.match_filenames_once()返回的文件列表作为临时变量并没有保存到checkpoint，
        # 所以并不会作为全局变量被global_variables_initializer()函数初始化，
        # 所以要进行局部变量初始化，不然会报错
        tf.local_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            for i in range(TRAINING_STEPS):
                # while not coord.should_stop():
                if coord.should_stop():
                    break
                _, loss_value, accu, step = sess.run([train_op, loss, accuracy, global_step])
                # 每迭代训练1000次，输出总损失和在训练样本上的分类正确率
                if i % 2 == 0:
                    # print(weightss)
                    print("After %d training step(s),loss on training batch is %g,accuracy is %g." % (
                        step, loss_value, accu))

                # 保存训练结束后的神经网络模型，在测试或者离线时，直接加载模型
                if i % 1000 == 0:
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
        coord.join(threads)


# 主程序入口
def main(argv=None):
    # 声明处理MNIST数据集的类，这个类在初始化时会自动下载数据
    # mnist=input_data.read_data_sets("/tmp/data/",one_hot=True)
    train()


# Tensorflow提供的一个主程序入口，tf.app.run会调用上面定义的main函数。
if __name__ == '__main__':
    tf.app.run()
