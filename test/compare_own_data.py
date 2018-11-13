#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/7 17:45
# @Author  : NYY
# @Site    : www.niuyuanyuanna.git.io
# @File    : compare_own_data.py
import os
import json
from config.configs import config
from sklearn.metrics import confusion_matrix


def save_labels(file_path, output_file_path):
    with open(file_path, 'rb') as f:
        for line in f:
            line = line.decode()
            line = line.split('\t')
            file_path = line[0]
            emotion = line[-2]
            emotion = emotion.replace('\'', '\"')
            emotion_j = json.loads(emotion)
            gt_emotion = file_path.split('\\')[-2]

            face_score = sorted(list(emotion_j.items()), key=lambda r: r[1], reverse=True)
            image_label_predict, image_predict_score = face_score[0]

            with open(output_file_path, 'a+') as fw:
                fw.write(gt_emotion + '\t' + image_label_predict)
                fw.write('\n')


def read_output(output_file):
    count = 0
    right_count = 0
    gt_result_dict = {}
    api_result_dict = {}
    with open(output_file, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            if line[0] in gt_result_dict:
                gt_result_dict[line[0]] += 1
            else:
                gt_result_dict[line[0]] = 1

            if line[1] in api_result_dict:
                api_result_dict[line[1]] += 1
            else:
                api_result_dict[line[1]] = 1

        print(gt_result_dict)
        print(api_result_dict)


def show_confusion_matrix(output_file, gt_dict, api_dict):
    gt_list = []
    api_list = []
    with open(output_file, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            gt_list.append(gt_dict[line[0]])
            api_list.append(api_dict[line[-1]])
    cm = confusion_matrix(gt_list, api_list)
    print(cm)


if __name__ == '__main__':
    gt_emotion_dict = {'amazing': 1, 'angry': 2, 'fear': 3, 'happy': 4, 'normal': 5, 'sad': 6, 'sick': 7}
    api_emotion_dict = {'surprise': 1, 'anger': 2,  'fear': 3, 'happiness': 4, 'neutral': 5, 'sadness': 6, 'disgust': 7}
    txt_base_path = os.path.join(config.data_root_path, 'own/logs')
    file_path = os.path.join(txt_base_path, 'exp_normal_image_list.txt')
    output_file_path = os.path.join(txt_base_path, 'compare.txt')
    # save_labels(file_path, output_file_path)
    # read_output(output_file_path)
    show_confusion_matrix(output_file_path, gt_emotion_dict, api_emotion_dict)





