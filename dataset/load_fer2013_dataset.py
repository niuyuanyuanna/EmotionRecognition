# -*- coding: utf-8 -*-
import os
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import glob


def take_second(elem):
    return elem.split(sep='_')[1]


def load_data_raw(filedir, train_image_dir, image_data_list, label, name_dict):
    train_image_list = os.listdir(filedir + '/' + train_image_dir)
    train_image_list.sort(key=take_second)
    for img in train_image_list:
        url = os.path.join(filedir, train_image_dir, img)
        image = load_img(url, grayscale=True, target_size=(48, 48))
        image_data_list.append(img_to_array(image))
        name = img.split('.')[0].split('_')
        # label.append(img.split('.')[0].split('_')[0])
        label.append(name_dict[name[0]])

    print('load' + train_image_dir + '%d images' % (len(train_image_list)))
    return image_data_list, label


def load_dataset(filedir, type):
    """
    读取数据
    :param filedir:
    :return:
    """
    image_data_list = []
    label = []
    name_dict = {'anger': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprised': 5, 'normal': 6}

    if type == 'train':
        image_data_list, label = load_data_raw(filedir, 'train_enhanced_sorted',
                                               image_data_list, label, name_dict)
    elif type == 'val':
        image_data_list, label = load_data_raw(filedir, 'val',
                                               image_data_list, label, name_dict)
    else:
        image_data_list, label = load_data_raw(filedir, 'test',
                                               image_data_list, label, name_dict)
    img_data = np.array(image_data_list)
    img_data = img_data.astype('float32')
    img_data /= 255
    return img_data, label


def load_imagePath_label_list(filedir):
    imagePath_list = []
    label_list = []
    extension = 'JPG'
    file_glob = os.path.join(filedir, '*.' + extension)
    file_paths = glob.glob(file_glob)
    for file_path in file_paths:
        dir_name = os.path.basename(file_path)
        label_name = dir_name.split('_')[0]
        imagePath_list.append(file_path)
        label_list.append(label_name)
    return imagePath_list, label_list


def load_all_imagePath_label_list(filedir):
    imagePath_list = []
    label_list = []
    subdirs = os.walk(filedir)
    for root, dirs, files in subdirs:
        for subdir in dirs:
            input_dir = os.path.join(filedir, subdir)
            img_list, l_list = load_imagePath_label_list(input_dir)
            imagePath_list.extend(img_list)
            label_list.extend(l_list)
    return imagePath_list, label_list