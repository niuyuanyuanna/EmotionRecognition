#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/12 14:49
# @Author  : NYY
# @Site    : www.niuyuanyuanna.git.io
# @File    : train_afn_dan.py
import os
import numpy as np
import tensorflow as tf

from model.tf_dan import emoDAN
from config.configs import config


def creat_file_path(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


if __name__ == '__main__':
    stage = 1
    s2_output_path = os.path.join(config.dataset.afn.model_path, 'model_s2')
    logging_file = os.path.join(config.dataset.afn.data_log, 'logging.txt')
    creat_file_path(config.dataset.afn.model_path)
    creat_file_path(config.dataset.afn.data_log)

    train_file_name = os.path.join(config.dataset.afn.csv_data, 'train_set.npz')
    val_file_name = os.path.join(config.dataset.afn.csv_data, 'val_set.npz')

    trainSet = np.load(train_file_name)
    validationSet = np.load(val_file_name)

    Xtrain = trainSet['Image']
    Ytrain = trainSet['Landmark']
    Ytrain_em = trainSet['Emotion']

    Xvalid = validationSet['Image'][:config.test.batch_size]
    Yvalid = validationSet['Landmark'][:config.test.batch_size]
    Yvalid_em = validationSet['Emotion'][:config.test.batch_size]

    nChannels = Xtrain.shape[3]
    nSamples = Xtrain.shape[0]
    testIdxsTrainSet = range(len(Xvalid))
    testIdxsValidSet = range(len(Xvalid))

    meanImg = trainSet['MeanShape']
    initLandmarks = trainSet['Landmark'][0].reshape((1, 136))

    with tf.Session() as sess:
        emotionaldan = emoDAN(initLandmarks, config.train.batch_size)

        Saver = tf.train.Saver()
        Writer = tf.summary.FileWriter(config.dataset.afn.data_log, sess.graph)

        if stage < 2:
            sess.run(tf.global_variables_initializer())
        else:
            pretrianed_model_path = 'pretrained_model_path'
            Saver.restore(sess, pretrianed_model_path)
            print('Pre-trained model has been loaded!')

        print("Starting training......")
        max_accuracy = 0
        min_err = 99999
        global_step = tf.Variable(0, trainable=False)

        for epoch in range(config.epoch):
            Count = 0
            while Count * config.train.batch_size < Xtrain.shape[0]:

                RandomIdx = np.random.choice(Xtrain.shape[0], config.train.batch_size, False)
                if stage == 1 or stage == 0:
                    # Training landmarks
                    sess.run(
                        emotionaldan['S1_Optimizer'],
                        feed_dict={emotionaldan['InputImage']: Xtrain[RandomIdx],
                                   emotionaldan['GroundTruth']: Ytrain[RandomIdx],
                                   emotionaldan['Emotion_labels']: Ytrain_em[RandomIdx],
                                   emotionaldan['S1_isTrain']: True,
                                   emotionaldan['S2_isTrain']: False,
                                   # emotionaldan['lr_stage2']:learning_rate
                                   })
                else:
                    # Training emotions
                    sess.run(
                        [emotionaldan['S2_Optimizer'], emotionaldan['iterator'].initializer],
                        feed_dict={emotionaldan['x']: Xtrain,
                                   emotionaldan['y']: Ytrain,
                                   emotionaldan['z']: Ytrain_em,
                                   emotionaldan['S1_isTrain']: False,
                                   emotionaldan['S2_isTrain']: True,
                                   # emotionaldan['lr_stage2']:learning_rate
                                   })

                if Count % 256 == 0:
                    TestErr = 0
                    BatchErr = 0

                    if stage == 1 or stage == 0:
                        # Validation landmarks
                        TestErr = sess.run(
                            emotionaldan['S1_Cost'],
                            {emotionaldan['InputImage']: Xvalid,
                             emotionaldan['GroundTruth']: Yvalid,
                             emotionaldan['Emotion_labels']: Yvalid_em,
                             emotionaldan['S1_isTrain']: False,
                             emotionaldan['S2_isTrain']: False,
                             # emotionaldan['lr_stage2']:learning_rate
                             })
                        BatchErr = sess.run(
                            emotionaldan['S1_Cost'],
                            {emotionaldan['InputImage']: Xtrain[RandomIdx],
                             emotionaldan['GroundTruth']: Ytrain[RandomIdx],
                             emotionaldan['Emotion_labels']: Ytrain_em[RandomIdx],
                             emotionaldan['S1_isTrain']: False,
                             emotionaldan['S2_isTrain']: False,
                             # emotionaldan['lr_stage2']:learning_rate
                             })
                        print('Epoch: ', epoch, ' Batch: ', Count,
                              'TestErr:', TestErr, ' BatchErr:', BatchErr)
                        if TestErr < min_err:
                            Saver.save(sess, s2_output_path)
                            min_err = TestErr
                    else:
                        # Validation emotions
                        TestErr, accuracy_test = sess.run(
                            [emotionaldan['Joint_Cost'],
                                emotionaldan['Emotion_Accuracy']],
                            {emotionaldan['x']: Xvalid,
                             emotionaldan['y']: Yvalid,
                             emotionaldan['z']: Yvalid_em,
                             emotionaldan['S1_isTrain']: False,
                             emotionaldan['S2_isTrain']: False})

                        BatchErr, accuracy_train, learn_rate = sess.run(
                            [emotionaldan['Joint_Cost'], emotionaldan['Emotion_Accuracy'], emotionaldan['lr']],
                            {emotionaldan['x']: Xtrain,
                             emotionaldan['y']: Ytrain,
                             emotionaldan['z']: Ytrain_em,
                             emotionaldan['S1_isTrain']: False,
                             emotionaldan['S2_isTrain']: False})
                        announce = 'Epoch: ' + str(epoch) + ' Batch: ' + str(Count) + ' TestErr: ' + str(TestErr) + ' BatchErr: ' + str(
                            BatchErr) + ' TestAcc: ' + str(accuracy_test) + ' TrainAcc: ' + str(accuracy_train) + ' LR: ' + str(learn_rate) + '\n'
                        print(announce)
                        with open(logging_file, 'a') as my_file:
                            my_file.write(announce)
                        if accuracy_test > max_accuracy:
                            Saver.save(sess, s2_output_path)
                            max_accuracy = accuracy_test
                Count += 1