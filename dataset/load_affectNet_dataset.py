# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/26 16:08
# @Author  : NYY
# @Site    : www.niuyuanyuanna.git.io
# @File    : load_affectNet_dataset.py
import os
import random
import cv2
import numpy as np

from config.configs import config


def load_filename_list(txt_file):
    file_list = []
    emotion_list = []
    landmark_list = []
    bbox_list = []
    with open(txt_file, 'rb') as fid:
        for line in fid:
            line = line.decode()
            line = line.strip()
            raw = line.split('\t')
            file_name = raw[0]
            bbox = raw[1]
            landmarks = raw[2]
            emotion = int(raw[-1])
            file_list.append(file_name)
            emotion_list.append(emotion)
            landmark_list.append(landmarks)
            bbox_list.append(bbox)
        print('load filelist down, lenght is %d' % len(file_list))
    return file_list, bbox_list, landmark_list, emotion_list


def analyse_dataset(emotion_labels):
    emotion_dict = {}
    for label in emotion_labels:
        if not label in emotion_dict:
            emotion_dict[label] = 0
        emotion_dict[label] += 1
    return emotion_dict


def formate_training_list(emotion_list):
    label_dict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
    output_dict = label_dict
    for i, emotion in enumerate(emotion_list):
        emotion = int(emotion)
        label_dict[emotion].append(i)

    for key in label_dict:
        print(key)
        max_range = len(label_dict[key])
        range_param = min(max_range, 5000)
        print(range_param)
        output_dict[key] = random.sample(label_dict[key], range_param)
    return output_dict


def save_training_file(output_dict, output_filepath,
                       file_list, bbox_list, landmark_list, emotion_list):
    filted_file_list = []
    filted_landmark_list = []
    filted_bbox_list = []
    filted_emotion_list = []
    for i, file_name in enumerate(file_list):
        emotion = emotion_list[i]
        with open(output_filepath, 'a') as fout:
            if i in output_dict[emotion]:
                bbox_xywh = bbox_list[i]
                landmark_i = landmark_list[i]

                filted_file_list.append(file_name)
                filted_landmark_list.append(landmark_i)
                filted_bbox_list.append(bbox_xywh)
                filted_emotion_list.append(emotion)

                fout.write(file_name + '\t' + bbox_xywh + '\t' + landmark_i + '\t' + emotion)
    return filted_file_list, filted_bbox_list, filted_landmark_list, filted_emotion_list


def formate_dataset_npz(file_list, landmark_list, emotion_labels, output_filepath):
    images_list = []
    landmarks_list = []
    emotions_list = []
    for i, file_path in enumerate(file_list):
        img = cv2.imread(file_path)
        img = np.asarray(img)

        landmarks_i = landmark_list[i].split(';')
        for j, point in enumerate(landmarks_i):
            landmarks_i[j] = float(point)
        landmarks_i = np.reshape(landmarks_i, (-1, 2))
        landmarks_i = np.asarray(landmarks_i)
        w_scale = img.shape[0] / float(config.dataset.afn.img_size)
        h_scale = img.shape[1] / float(config.dataset.afn.img_size)
        landmarks_i[:, 0] = landmarks_i[:, 0] / w_scale
        landmarks_i[:, 1] = landmarks_i[:, 1] / h_scale
        landmarks_i = landmarks_i.flatten()
        img = cv2.resize(img, (config.dataset.afn.img_size, config.dataset.afn.img_size))
        emotion_i = int(emotion_labels[i])

        images_list.append(img)
        landmarks_list.append(landmarks_i)
        emotions_list.append(emotion_i)
    np.savez(output_filepath, images_list, landmarks_list, emotions_list)


if __name__ == '__main__':
    origin_txt_file = os.path.join(config.dataset.afn.csv_data, 'train_c.txt')
    file_list, bbox_list, landmark_list, emotion_list = load_filename_list(origin_txt_file)
    emotion_dict = analyse_dataset(emotion_list)
    print('analyze train_cleaned txt file done')
    print(emotion_dict)

    train_output_file = os.path.join(config.dataset.afn.csv_data, 'train_filted.txt')
    output_dict = formate_training_list(emotion_list)
    filted_file_list, filted_bbox_list, \
    filted_landmark_list, filted_emotion_list = save_training_file(output_dict, train_output_file,
                                                                   file_list, bbox_list, landmark_list, emotion_list)
    print('save train_filted txt file done')

    npz_file_path = os.path.join(config.dataset.afn.csv_data, 'train.npz')
    formate_dataset_npz(filted_file_list, filted_landmark_list, filted_emotion_list, npz_file_path)
    print('save train_filted npz file done')

    val_txt_file = os.path.join(config.dataset.afn.csv_data, 'val_c.txt')
    val_file_list, val_bbox_list, val_landmark_list, val_emotion_list = load_filename_list(val_txt_file)
    emotion_dict = analyse_dataset(val_emotion_list)
    print('analyze val_cleaned txt file done')
    print(emotion_dict)

