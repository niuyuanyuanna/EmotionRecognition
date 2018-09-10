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
import shutil
import time


from dataset.load_fer2013_dataset import load_all_imagePath_label_list
''''终结点: https://westcentralus.api.cognitive.microsoft.com/face/v1.0

密钥 1: dc1104cfaa1d4f18859f5fbd6584c599

密钥 2: 515a61302dc2408685b560b916d6b76e'''


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


def split_image_by_class(image_label_predict, image_path, clean_path):
    subfloder = os.path.join(clean_path, image_label_predict)
    if not os.path.exists(subfloder):
        os.makedirs(subfloder)
    new_iamge_name = os.path.basename(image_path)
    new_image_save_file = os.path.join(subfloder, new_iamge_name)
    shutil.copy(image_path, new_image_save_file)


def clean_fer_image(image_path, clean_path):
    subscription_key = '137de087e5004613b29d0c10419574c7'
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
    response.raise_for_status()
    faces = response.json()
    for face in faces:
        fa = face['faceAttributes']
        emotion = fa['emotion']
        face_score = sorted(list(emotion.items()), key=lambda r: r[1], reverse=True)
        image_label_predict = 'negative'
        for key, value in face_score:
            if value > 0.7:
                image_label_predict = key
        split_image_by_class(image_label_predict, image_path, clean_path)


def add_blank_to_image(imagePath_list, label_list, save_dir):
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


if __name__ == '__main__':
    database_path = 'E:/liuyuan/DataCenter/DatasetFER2013/fer2013'
    csv_file = os.path.join(database_path, 'fer2013.csv')
    save_path = os.path.join(database_path, 'rebuild_images')
    addBlank_path = os.path.join(database_path, 'padding_images')
    clean_path = os.path.join(database_path, 'cleaned_images')
    label_dict = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
    # save_data_as_images(csv_file, save_path, label_dict)
    # imagePath_list, label_list = load_all_imagePath_label_list(save_path)
    # # padding images
    # add_blank_to_image(imagePath_list, label_list, addBlank_path)

    padidng_image_path_list, padding_label_list = load_all_imagePath_label_list(addBlank_path)
    # use microsoft api clean images
    for i in range(len(padding_label_list)):
        if i >= 4:
            clean_fer_image(padding_label_list[i], clean_path)
            time.sleep(15)


