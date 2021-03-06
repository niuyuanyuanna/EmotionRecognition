#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/7 14:44
# @Author  : NYY
# @Site    : www.niuyuanyuanna.git.io
# @File    : clean_fer_dataset.py
import csv
import os
import numpy as np
from PIL import Image
import requests
import time
import json

from dataset.load_fer2013_dataset import load_all_imagePath_label_list
from config.configs import config


def save_data_as_images(csv_file, save_path, label_dict):
    with open(csv_file) as f:
        csvr = csv.reader(f)
        header = next(csvr)
        for i, (label, pixel, usage) in enumerate(csvr):
            pixel = np.asarray([float(p) for p in pixel.split()]).reshape(48, 48)
            subfolder = os.path.join(save_path, label_dict[int(label)])
            if not os.path.exists(subfolder):
                os.makedirs(subfolder)
            im = Image.fromarray(pixel).convert('L')
            image_name = os.path.join(subfolder, '{}_{:05d}.jpg'.format(label_dict[int(label)], i))
            print(image_name)
            im.save(image_name)


def padding_image(imagePath_list, label_list, save_dir):
    for i, imagePath in enumerate(imagePath_list):
        old_image = Image.open(imagePath)
        new_image = Image.new('L', (128, 128))
        new_image.paste(old_image, (40, 40))
        image_name = os.path.basename(imagePath)

        saved_dir = os.path.join(save_dir, label_list[i])
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)

        save_image_path = os.path.join(saved_dir, image_name)
        new_image.save(save_image_path)
    print('format %d images' % len(imagePath_list))


def clean_fer_image(image_path, image_label, error_txt_file, norm_txt_file):
    subscription_key = '2780db0320144f5d885b55d4c3bc6427'
    emotion_recognition_url = "https://api.cognitive.azure.cn/face/v1.0/detect"
    headers = {'Ocp-Apim-Subscription-Key': subscription_key, 'Content-Type': 'application/octet-stream'}
    params = {
        'returnFaceId': 'true',
        'returnFaceLandmarks': 'false',
        'returnFaceAttributes': 'age,gender,smile,emotion'
    }
    data = open(image_path, 'rb')
    img_url = {'url': image_path}
    response = requests.post(emotion_recognition_url, params=params,
                             headers=headers, data=data, json=img_url)
    save_string = ''
    is_error = False
    if response.status_code != requests.codes.ok:
        error = response.json()
        error = error['error']
        error_code = error['code']
        is_error = True
        save_string = image_path + '\t' + image_label + '\t' + 'None' + '\t' + '0' + '\t' + error_code + '\n'
        with open(error_txt_file, 'a') as fe:
            fe.write(save_string)

    else:
        is_error = False
        faces = response.json()
        for face in faces:
            fa = face['faceAttributes']
            emotion = fa['emotion']
            face_score = sorted(list(emotion.items()), key=lambda r: r[1], reverse=True)
            image_label_predict, image_predict_score = face_score[0]
            face_score_all = json.dumps(emotion)
            save_string = image_path + '\t' + image_label + '\t' + image_label_predict + '\t' \
                          + str(image_predict_score) + '\t' + face_score_all + '\t' + 'error_code' + '\n'
            with open(norm_txt_file, 'a') as fn:
                fn.write(save_string)

    return save_string, is_error


def save_return_info(info_list, save_txt_path):
    info_file = open(save_txt_path, 'w')
    for norm_info in info_list:
        info_file.write(norm_info)
    info_file.close()


def creat_info_list(padidng_image_path_list, padding_label_list, error_txt_file, norm_txt_file):
    error_str_list = []
    norm_str_list = []
    for i in range(0, len(padding_label_list)):
        save_string, is_error = clean_fer_image(padidng_image_path_list[i], padding_label_list[i], error_txt_file,
                                                norm_txt_file)
        if is_error:
            error_str_list.append(save_string)
            print('error: %s' % padidng_image_path_list[i])

        else:
            norm_str_list.append(save_string)
            print('normal: %s' % padidng_image_path_list[i])
        time.sleep(5)
    return error_str_list, norm_str_list


# ========================= clean后发现部分数据未被清理 =================================
# =====需要将未被清理到的数据重新构建list，送入clean函数中================================

def load_diff_file(diff_file_path):
    diff_list = []
    diff_label = []
    with open(diff_file_path, 'rb') as fid:
        for line in fid:
            line = line.decode()
            line = line.strip()
            diff_list.append(line)

            image_name = os.path.basename(line)
            label = image_name.split('_')[0]
            diff_label.append(label)
    return diff_list, diff_label


def clean_diff_file(diff_list, diff_label, error_txt_file, norm_txt_file):
    for i in range(len(diff_list)):
        save_str, is_error = clean_fer_image(diff_list[i], diff_label[i], error_txt_file, norm_txt_file)
        print(i)
        if is_error:
            print('error: %s' % save_str)

        else:
            print('normal: %s' % save_str)
        time.sleep(5)


if __name__ == '__main__':
    csv_file = config.dataset.fer2013.origin_csv_file
    save_path = config.dataset.fer2013.rebuild_image_from_csv
    addBlank_path = config.dataset.fer2013.padding_image_path
    clean_path = config.dataset.fer2013.cleaned_face_imgs_path

    label_dict = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
    # save_data_as_images(csv_file, save_path, label_dict)
    # imagePath_list, label_list = load_all_imagePath_label_list(save_path)
    # # padding images
    # add_blank_to_image(imagePath_list, label_list, addBlank_path)

    # padidng_image_path_list, padding_label_list = load_all_imagePath_label_list(addBlank_path)
    # use microsoft api clean images
    # error_txt_file = os.path.join(clean_path, 'error_image.txt')
    # norm_txt_file = os.path.join(clean_path, 'norm_image.txt')
    # error_str_list, norm_str_list = creat_info_list(padidng_image_path_list, padding_label_list, error_txt_file, norm_txt_file)

    diff_error_txt_file = os.path.join(clean_path, 'diff_error_image.txt')
    diff_norm_txt_file = os.path.join(clean_path, 'diff_norm_image.txt')
    diff_file_path = os.path.join(clean_path, 'diff_list.txt')
    diff_list, diff_label = load_diff_file(diff_file_path)

    clean_diff_file(diff_list, diff_label, diff_error_txt_file, diff_norm_txt_file)
