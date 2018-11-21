#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/31 11:47
# @Author  : NYY
# @Site    : www.niuyuanyuanna.git.io
# @File    : clean_affectNet_dataset.py
import csv
import os
from PIL import Image

from config.configs import config


def clean_csv(input_filepath, output_filepath):
    with open(input_filepath, 'r') as fr:
        reader = csv.reader(fr)
        with open(output_filepath, 'a') as fe:
            for i, row in enumerate(reader):
                if reader.line_num == 1:
                    continue
                expression = row[6]
                if int(expression) >= 7:
                    continue
                subDirectory_filePath = row[0]
                image_full_path = os.path.join(config.dataset.afn.image_path, subDirectory_filePath)
                if os.path.exists(image_full_path):
                    try:
                        Image.open(image_full_path).load()
                        # img = cv2.imread(image_full_path)
                    except Exception as e:
                        print(Exception, ":", e)
                        continue

                    bbox_xywh = row[1:5]
                    bbox_xywh = ';'.join(bbox_xywh)
                    facial_landmarks = row[5]
                    fe.write(image_full_path + '\t' + bbox_xywh + '\t' + facial_landmarks + '\t' + expression)


if __name__ == '__main__':
    input_file = os.path.join(config.dataset.afn.csv_data, 'training.csv')
    output_file = os.path.join(config.dataset.afn.csv_data, 'train_c.txt')
    clean_csv(input_file, output_file)
    input_file = os.path.join(config.dataset.afn.csv_data, 'validation.csv')
    output_file = os.path.join(config.dataset.afn.csv_data, 'val_c.txt')
    clean_csv(input_file, output_file)
