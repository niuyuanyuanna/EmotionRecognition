#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/26 16:02
# @Author  : NYY
# @Site    : www.niuyuanyuanna.git.io
# @File    : train_afn_emotion.py
# train affectnet by mobilenet
import os
from random import shuffle

from model.cnn import mobileNetv2
from dataset.load_affectNet_dataset import load_filename_list
from config.configs import config
from util.image_aug import ImageGenerator

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.callbacks import CSVLogger, EarlyStopping


def creat_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def train():
    train_csv = os.path.join(config.dataset.afn.csv_data, 'train_cleaned.csv')
    train_image_names, train_image_labels = load_filename_list(train_csv)

    train_dicts = list(zip(train_image_names, train_image_labels))
    shuffle(train_dicts)
    train_image_names, train_image_labels = zip(*train_dicts)

    test_csv = os.path.join(config.dataset.afn.csv_data, 'val_cleaned.csv')
    test_image_names, test_image_labels = load_filename_list(test_csv)

    image_generator = ImageGenerator(train_image_names, train_image_labels, test_image_names, test_image_labels, config)

    model = mobileNetv2(config.dataset.input_resolution, num_classes=7)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    early_stop = EarlyStopping(monitor='val_acc', patience=30, verbose=0, mode='auto')
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(50 / 4), verbose=1)

    log_file_path = os.path.join(config.tmp.root_path, 'emotion_train_afn.log')
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

    model.fit_generator(
        image_generator.flow(mode='train'),
        validation_data=image_generator.flow('val'),
        steps_per_epoch=int(len(train_image_names) / config.train.batch_size),
        validation_steps=int(len(test_image_names) / config.train.batch_size),
        epochs=config.epoch,
        callbacks=callbacks)


if __name__ == '__main__':
    train()