#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/27 15:11
# @Author  : NYY
# @File    : test_fer2013_tf_LeNet.py
# @Software: PyCharm
import sys
sys.path.append()
import tensorflow as tf


from model.leNet_inference import inference
from dataset.format_batch import get_batch
import train_fer2013_tf_LeNet as trainLe
from config.configs import config


# TFRecord文件存储地址
filename = config.fer2013_test_TFRecord_file_path


def evaluate():
    with tf.Graph().as_default() as g:
        xs, ys = get_batch(filename)
        global_step = tf.Variable(0, trainable=False)

        # 直接通过调用封装好的函数来计算前向传播结果。因为测试时不关注正则化损失的，所以这里用于计算正则化损失的函数被
        # 设置为None,并且train设置为False。
        y = inference(xs, False, None)

        # 使用前向传播的结果计算正确率。如果需要对未知样例进行分类，那么使用tf.argmax(y,1)就可以得到输入样例的预测类别了
        correct_prediction = tf.equal(tf.argmax(y, 1), ys)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 通过变量重命名的方式来加载模型，这样在前向传播的过程中就不需要调用求滑动平均的函数来获取平均值了。这样就可以完全共用
        # mnsit_inference.py中定义前向传播过程。
        variable_averages = tf.train.ExponentialMovingAverage(trainLe.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

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
                # tf.train.get_checkpoint_state函数会通过cheakpoint文件自动找目录中最新模型的文件名。
                ckpt = tf.train.get_checkpoint_state(trainLe.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 通过文件名得到模型保存时迭代的轮数（训练迭代的总轮数）
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    # 计算测试数据分类正确率
                    accuracy_test = sess.run(accuracy)
                    # 输出测试结果和验证结果
                    print("After %s training step(s), test accuracy=%g" % (global_step, accuracy_test))
            except tf.errors.OutOfRangeError:
                print('done!')
            finally:
                coord.request_stop()
            coord.join(threads)


# 主程序入口
def main(argv=None):
    # 声明处理MNIST数据集的类，这个类在初始化时会自动下载数据
    evaluate()


# TensorFlow提供的一个主程序入口，tf.app.run会调用上面定义的main函数。
if __name__ == '__main__':
    tf.app.run()
