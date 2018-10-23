#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/15 14:14
# @Author  : NYY
# @Site    : www.niuyuanyuanna.git.io
# @File    : clean_own_dataset.py
"""
the usage of this file is to clean human faces which are collected by our own by using face++ emotion
recognition API to classify the natual and happy emotion.
API Key: hTQc_PYcFerIsTqAoalCUSNBNrvfl6j4
API Secret: iwe_88vYqh_XRR8s9nEYhAo3bwxxhoeZ
"""
import os
import requests
import time
from json import JSONDecoder

from config.configs import config


def get_origin_image_list():
    own_dataset_root_path = os.path.join(config.data_root_path, 'own/cutSmallFace')
    images_list = []
    subdirs = os.walk(own_dataset_root_path)
    for root, dirs, _ in subdirs:
        for sub_sub_dir in dirs:   # sub_sub_dir对应文件的人名，如liuyuan
            temp_path = os.path.join(root, sub_sub_dir)
            for detail_root, detail_dirs, _ in os.walk(temp_path):
                # detail_dirs = ['nameleft', 'nameright', 'namemid']
                for detail_dir in detail_dirs:
                    temp_detail_path = os.path.join(detail_root, detail_dir)
                    print(temp_detail_path)
                    for final_root, _, final_files in os.walk(temp_detail_path):
                        for final_file in final_files:
                            final_image_path = os.path.join(temp_detail_path, final_file)
                            images_list.append(final_image_path)
    return images_list


def creat_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return


def write_list_to_file(file_path, images_list):
    with open(file_path, 'w') as f:
        for image_path in images_list:
            f.write(image_path + '\n')
    return


def load_images_list(txt_path):
    images_list = []
    with open(txt_path, 'rb') as fid:
        for line in fid:
            line = line.decode()
            line = line.strip()
            images_list.append(line)
    return images_list


def save_face_token(filepath, normal_txt_path, error_txt_path):
    http_url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
    key = 'hTQc_PYcFerIsTqAoalCUSNBNrvfl6j4'
    secret = 'iwe_88vYqh_XRR8s9nEYhAo3bwxxhoeZ'
    data = {"api_key": key, "api_secret": secret, "return_landmark": "2", "return_attributes": "emotion"}
    files = {"image_file": open(filepath, "rb")}

    response = requests.post(http_url, data=data, files=files)
    if response.status_code != requests.codes.ok:
        error = response.json()
        error = error['error_message']
        save_str = filepath + '\t' + '-1' + '\t' + '-1' + '\t' + '-1' +'\t' + error + '\n'
        print('image error message: %s' % error)
        with open(error_txt_path, 'a+') as f:
            f.write(save_str)
            return
    else:

        req_con = response.content.decode('utf-8')
        req_dict = JSONDecoder().decode(req_con)
        faces = req_dict['faces']
        if faces is None or len(faces) == 0:
            save_str = filepath + '\t' + '-1' + '\t' + '-1' + '\t' + '-1' +'\t' + '-1' + '\n'
            print('image error, no face detected')
            with open(error_txt_path, 'a+') as f:
                f.write(save_str)
                return
        face = faces[0]


        face_token = face['face_token']
        face_landmark = face['landmark']
        face_attributes = face['attributes']
        face_emotion = face_attributes['emotion']
        save_str = filepath + '\t' + face_token + '\t' + str(face_landmark) + '\t' + str(face_emotion) + '\t' + '200' + '\n'
        print('save right image %s' % filepath)
        with open(normal_txt_path, 'a+') as f:
            f.write(save_str)
    return


def clean_images(images_list, normal_txt_path, error_txt_path):
    for i in range(46968, len(images_list)):
        image_path = images_list[i]
        print('deal with %d th image' % i)
        save_face_token(image_path, normal_txt_path, error_txt_path)
        time.sleep(3)


if __name__ == '__main__':
    # images_list = get_origin_image_list()

    txt_base_path = os.path.join(config.data_root_path, 'own/logs')
    txt_full_path = os.path.join(txt_base_path, 'origin_image_lists.txt')
    # write_list_to_file(txt_full_path, images_list)

    images_list = load_images_list(txt_full_path)
    normal_txt_path = os.path.join(txt_base_path, 'normal_image_list.txt')
    error_txt_path = os.path.join(txt_base_path, 'error_image_list.txt')
    clean_images(images_list, normal_txt_path, error_txt_path)






