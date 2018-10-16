#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/8 14:46
# @Author  : NYY
# @Site    : www.niuyuanyuanna.git.io
# @File    : train_emotionnet.py
import os

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from random import shuffle

from dataset.load_raf_dataset import load_normal_list
from util.image_aug import ImageGenerator
from config.configs import config
from model.cnn import mini_XCEPTION


def creat_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def train():
    train_txt = os.path.join(config.dataset.raf.label_list_path, 'train_image_label.txt')
    test_txt = os.path.join(config.dataset.raf.label_list_path, 'test_image_label.txt')
    train_image_names, train_image_labels = load_normal_list(train_txt)
    test_image_names, test_image_labels = load_normal_list(test_txt)
    train_dicts = list(zip(train_image_names, train_image_labels))
    shuffle(train_dicts)
    train_image_names, train_image_labels = zip(*train_dicts)

    image_generator = ImageGenerator(train_image_names, train_image_labels, test_image_names, test_image_labels, config)

    model = mini_XCEPTION(input_shape=(100, 100, 3), num_classes=7)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    early_stop = EarlyStopping('val_loss', patience=100)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                  patience=int(50 / 4), verbose=1)

    log_file_path = os.path.join(config.tmp.root_path, 'emotion_train.log')
    trained_model_path = config.model.tmp_kerase_model_save_path
    creat_path(trained_model_path)
    creat_path(config.tmp.root_path)

    csv_logger = CSVLogger(log_file_path, append=False)
    model_names = os.path.join(trained_model_path, '.{epoch:02d}-{val_acc:.2f}.hdf5')
    model_checkpoint = ModelCheckpoint(model_names,
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=False)
    callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

    # training model
    model.fit_generator(image_generator.flow(mode='train'),
                        steps_per_epoch=int(len(train_image_names) / config.train.batch_size),
                        epochs=config.epoch, verbose=1,
                        callbacks=callbacks,
                        validation_data=image_generator.flow('val'),
                        validation_steps=int(len(test_image_names) / config.train.batch_size))


if __name__ == '__main__':
    train()