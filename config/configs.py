#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/29 9:21
# @Author  : NYY
# @File    : configs.py
# @Software: PyCharm
from easydict import EasyDict as edict
import os


config = edict()
config.data_root_path = 'E:/liuyuan/DataCenter'

config.dataset = edict()
config.dataset.ck = edict()
config.dataset.fer2013 = edict()
config.dataset.raf = edict()
config.model = edict()
config.tmp = edict()

config.dataset.input_resolution = [28, 28]


config.dataset.ck.data_path = os.path.join(config.data_root_path, 'DatasetCK+')
config.dataset.ck.origin_img_path = os.path.join(config.dataset.ck.data_path, 'origin_imgs')
config.dataset.ck.label_data_path = os.path.join(config.dataset.ck.data_path, 'origin_labels')
config.dataset.ck.enhanced_img_path = os.path.join(config.dataset.ck.data_path, 'image')
config.dataset.ck.data_bottleneck_path = os.path.join(config.dataset.ck.data_path, 'bottleneck')
config.dataset.ck.total_train_img_path = os.path.join(config.dataset.ck.data_path, 'enhance_and_sorted_face_imgs')

config.dataset.ck.tfRecord_path = os.path.join(config.dataset.ck.data_path, 'TFRecord')
config.dataset.ck.train_TFRecord_file_path = os.path.join(config.dataset.ck.tfRecord_path, 'train.tfrecords')
config.dataset.ck.test_TFRecord_file_path = os.path.join(config.dataset.ck.tfRecord_path, 'test.tfrecords')
config.dataset.ck.val_TFRecord_file_path = os.path.join(config.dataset.ck.tfRecord_path, 'val.tfrecords')
config.dataset.ck.total_image_TFRecord_file_path = os.path.join(config.dataset.ck.tfRecord_path, 'train_total.tfrecords')
config.dataset.ck.total_train_TFRecord_file_path = os.path.join(config.dataset.ck.tfRecord_path, 'train_total.tfrecords')

config.dataset.fer2013.data_path = os.path.join(config.data_root_path, 'DatasetFER2013/face_imgs')
config.dataset.fer2013.origion_path = os.path.join(config.dataset.fer2013.data_path, 'train_origion')
config.dataset.fer2013.test_data_file = os.path.join(config.dataset.fer2013.data_path, 'test')
config.dataset.fer2013.val_data_file = os.path.join(config.dataset.fer2013.data_path, 'val')
config.dataset.fer2013.total_train_data_file = os.path.join(config.dataset.fer2013.data_path, 'train_enhanced_sorted')

config.dataset.fer2013.data_path_new = os.path.join(config.data_root_path, 'DatasetFER2013/fer2013')
config.dataset.fer2013.cleaned_face_imgs_path = os.path.join(config.dataset.fer2013.data_path_new, 'cleaned_images')
config.dataset.fer2013.origin_csv_file = os.path.join(config.dataset.fer2013.data_path_new, 'fer2013.csv')
config.dataset.fer2013.rebuild_image_from_csv = os.path.join(config.dataset.fer2013.data_path_new, 'rebuild_images')
config.dataset.fer2013.padding_image_path = os.path.join(config.dataset.fer2013.data_path_new, 'padding_images')
config.dataset.fer2013.face_image_path = os.path.join(config.dataset.fer2013.data_path_new, 'face_images')

config.dataset.fer2013.tfRecord_path = os.path.join(config.dataset.fer2013.data_path, 'TFRecord')
config.dataset.fer2013.train_TFRecord_file_path = os.path.join(config.dataset.fer2013.tfRecord_path, 'train.tfrecords')
config.dataset.fer2013.test_TFRecord_file_path = os.path.join(config.dataset.fer2013.tfRecord_path, 'test.tfrecords')
config.dataset.fer2013.val_TFRecord_file_path = os.path.join(config.dataset.fer2013.tfRecord_path, 'val.tfrecords')
config.dataset.fer2013.total_image_TFRecord_file_path = os.path.join(config.dataset.fer2013.tfRecord_path, 'train_total.tfrecords')
config.dataset.fer2013.total_train_TFRecord_file_path = os.path.join(config.dataset.fer2013.tfRecord_path, 'train_total.tfrecords')

config.dataset.raf.RAF_path = os.path.join(config.data_root_path, 'RAF-DB')
config.dataset.raf.aligned_image_path = os.path.join(config.dataset.raf.RAF_path, 'Image/aligned')
config.dataset.raf.label_list_path = os.path.join(config.dataset.raf.RAF_path, 'EmoLabel')
config.dataset.raf.b_g_r_mean = [108.74949161620336, 121.93682191782138, 129.882933212282]
config.dataset.raf.b_g_r_std = [76.27990887272082, 72.2227214967396, 73.87829834580016]

config.model.root_path = os.path.join(config.data_root_path, 'Models')
config.model.inception_tf_model = os.path.join(config.model.root_path, 'Inception-V3')
config.model.tmp_model_save_path = os.path.join(config.model.root_path, 'emotion_models')
config.model.tmp_tf_model_save_path = os.path.join(config.model.tmp_model_save_path, 'tensorflow')
config.model.tmp_kerase_model_save_path = os.path.join(config.model.tmp_model_save_path, 'keras')

config.tmp.root_path = os.path.join(config.data_root_path, 'EmotionLog')
config.tmp.model_graph = os.path.join(config.tmp.root_path, 'graph_path')



# train detail
config.train = edict()
config.train.split_val = 0.2
config.train.aug_strategy = edict()
config.train.aug_strategy.resize = True
config.train.aug_strategy.resize_size = 64
config.train.aug_strategy.normalize = True
config.train.aug_strategy.flip = True
config.train.aug_strategy.random_rotate = True
config.train.aug_strategy.random_crop = False
config.train.aug_strategy.random_color = False
config.train.aug_strategy.max_rotate_angle = 20

# test
config.test = edict()
config.test.aug_strategy = edict()
config.test.aug_strategy.resize = True
config.test.aug_strategy.resize_size = 64
config.test.aug_strategy.normalize = True
config.test.aug_strategy.flip = False
config.test.aug_strategy.random_rotate = False
config.test.aug_strategy.random_crop = False
config.test.aug_strategy.random_color = False
config.test.aug_strategy.max_rotate_angle = 20

# model params
config.support_network = ['ResNet']
config.network = "ResNet#50"
assert config.network.split('#')[0] in config.support_network
config.out_classes = [24, 300]
config.epoch = 60
config.train.batch_size = 60
config.train.repeat = 1
config.test.batch_size = 1500
config.data_loader_num_workers = 8
config.num_gpu = 1

# optimizer params
config.momentum = 0.0
config.weightDecay = 0.0
config.alpha = 0.99
config.epsilon = 1e-8

config.lr_params = edict()
config.lr_params.lr = 0.00125 * config.num_gpu * config.train.batch_size
config.lr_params.lr_step = [40, 55]
config.lr_params.decay = 0.1
config.lr_params.warm_up = True
config.lr_params.warm_up_lr = 0.001
config.lr_params.warm_up_epoch = 2

config.sample_test = None
config.DEBUG = False

config.pre_params = edict()
config.pre_params.window = True
config.pre_params.threshold = 0.0
config.pre_params.ignore_multi = True
config.pre_params.grow = 10
config.pre_params.min_proportion = 0.1  # 最小比例