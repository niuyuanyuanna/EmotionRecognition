# -*- coding: utf-8 -*-
import numpy as np
from model.keras_models import *
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, TensorBoard
from datetime import datetime
from time import time
import os
import sys
current_path = os.getcwd()
sys.path.append(current_path)

from dataset.load_fer2013_dataset import load_dataset


if __name__ == '__main__':
    train_loss = []
    train_accuracy = []
    file_dir = config.fer2013_data_path
    tmp_model_path = config.tmp_kerase_model_save_path

    log_dir = datetime.now().strftime(os.path.join(tmp_model_path, 'model_%Y-%m-%d_%H%M'))
    os.mkdir(log_dir)

    model_checkpoint = ModelCheckpoint(log_dir + '\\CIFAR10-EP{epoch:02d}-ACC{val_acc:.4f}.h5',
                                       monitor='val_acc', save_best_only=True)
    tensor_board = TensorBoard(log_dir=log_dir, histogram_freq=0)

    train_x, train_y = load_dataset(file_dir, 'train')
    val_x, val_y = load_dataset(file_dir, 'val')
    test_x, test_y = load_dataset(file_dir, 'test')

    train_y = np_utils.to_categorical(train_y)
    val_y = np_utils.to_categorical(val_y)
    test_y = np_utils.to_categorical(test_y)

    model = make_basic_network(0.5)
    start_time = time()
    h = model.fit(train_x, train_y, batch_size=128, epochs=400, verbose=1,
                  validation_data=(val_x, val_y),
                  callbacks=[model_checkpoint, tensor_board])
    model.save(os.path.join(tmp_model_path, 'model-basic_v1.h5'))
    end_time = time()
    print('\n@ Total Time Spent: %.2f seconds' % (end_time - start_time))
    acc, val_acc = h.history['acc'], h.history['val_acc']
    m_acc, m_val_acc = np.argmax(acc), np.argmax(val_acc)
    print("@ Best Training Accuracy: %.2f %% achieved at EP #%d." % (acc[m_acc] * 100, m_acc + 1))
    print("@ Best Testing Accuracy: %.2f %% achieved at EP #%d." % (val_acc[m_val_acc] * 100, m_val_acc + 1))

    # 评估模型
    loss, accuracy = model.evaluate(test_x, test_y)
    print('loss:', loss)
    print('accuracy:', accuracy)



