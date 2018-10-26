#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/26 16:08
# @Author  : NYY
# @Site    : www.niuyuanyuanna.git.io
# @File    : load_affectNet_dataset.py
import csv
import pandas as pd


def read_csv_file(file_path):
    with open(file_path) as f:
        csvr = csv.reader(f)
        header = next(csvr)
        for i, (file_path, face_x, face_y, face_width, face_height, facial_landmarks, expression, valence, arousal) in enumerate(csvr):
            pixel = np.asarray([float(p) for p in pixel.split()]).reshape(48, 48)
            subfolder = os.path.join(save_path, label_dict[int(label)])
            if not os.path.exists(subfolder):
                os.makedirs(subfolder)
            im = Image.fromarray(pixel).convert('L')
            image_name = os.path.join(subfolder, '{}_{:05d}.jpg'.format(label_dict[int(label)], i))
            print(image_name)
            im.save(image_name)
