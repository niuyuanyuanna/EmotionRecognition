# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/26 16:08
# @Author  : NYY
# @Site    : www.niuyuanyuanna.git.io
# @File    : load_affectNet_dataset.py
import csv
import os
import random


from config.configs import config


def load_filename_list(csv_file):
    file_list = []
    emotion_labels = []
    landmarks_list = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f, dialect=csv.excel_tab)
        for raw in reader:
            file_name = raw[0]
            landmarks = raw[2].replace('[', '')
            landmarks = landmarks.replace(']', '')
            landmarks = landmarks.split(',')
            for i in range(len(landmarks)):
                landmarks[i] = float(landmarks[i])
            emotion = int(raw[-1])
            file_list.append(file_name)
            emotion_labels.append(emotion)
            landmarks_list.append(landmarks)
    return file_list, landmarks_list, emotion_labels


def analyse_dataset(emotion_labels):
    emotion_dict = {}
    for label in emotion_labels:
        if not label in emotion_dict:
            emotion_dict[label] = 0
        emotion_dict[label] += 1
    return emotion_dict


def formate_training_list(csv_file):
    label_dict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
    output_dict = label_dict
    with open(csv_file, 'rb') as f:
        reader = csv.reader(f, dialect=csv.excel_tab)
        for i, raw in enumerate(reader):
            emotion = int(raw[-1])
            label_dict[emotion].append(i)

    for key in label_dict:
        print(key)
        max_range = len(label_dict[key])
        range_param = min(max_range, 5000)
        print(range_param)
        output_dict[key] = random.sample(label_dict[key], range_param)
    return output_dict


def save_training_file(output_dict, csv_file, output_filepath):
    with open(csv_file, 'r') as f:
        reader = csv.reader(f, dialect=csv.excel_tab)
        with open(output_filepath, 'wb+') as fw:
            writer = csv.writer(fw)
            for i, raw in enumerate(reader):
                image_full_path = raw[0]
                bbox_xywh = raw[1]
                facial_landmarks = raw[2]
                emotion = int(raw[-1])
                if i in output_dict[emotion]:
                    writer.writerow((image_full_path, bbox_xywh, facial_landmarks, emotion))


if __name__ == '__main__':
    csv_file = os.path.join(config.dataset.afn.csv_data, 'train_cleaned_server.csv')
    file_list, _, emotion_labels = load_filename_list(csv_file)
    emotion_dict = analyse_dataset(emotion_labels)
    print(emotion_dict)

    output_file = os.path.join(config.dataset.afn.csv_data, 'train_filted.csv')
    output_dict = formate_training_list(csv_file)
    save_training_file(output_dict, csv_file, output_file)
    # load_filename_list(output_file)
    print('load train csv file done')

    csv_file = os.path.join(config.dataset.afn.csv_data, 'val_cleaned1.csv')
    val_file_list, _, emotion_labels = load_filename_list(csv_file)
    emotion_dict = analyse_dataset(emotion_labels)
    print(emotion_dict)
