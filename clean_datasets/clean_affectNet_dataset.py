#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/31 11:47
# @Author  : NYY
# @Site    : www.niuyuanyuanna.git.io
# @File    : clean_affectNet_dataset.py
import csv
import os

from config.configs import config


def clean_csv(input_filepath, output_filepath):
    with open(input_filepath, 'r') as fr:
        reader = csv.reader(fr)
        fw = open(output_filepath, 'w', newline ='')
        try:
            writer = csv.writer(fw)
            for i, row in enumerate(reader):
                if reader.line_num == 1:
                    continue
                expression = row[6]
                if expression == '8' or expression == '10':
                    print('the %d th image continue, with emotion %s' % (i, expression))
                    continue

                subDirectory_filePath = row[0]
                image_full_path = os.path.join(config.dataset.afn.image_path, subDirectory_filePath)
                bbox_xywh = row[1:5]
                facial_landmarks = row[5]
                facial_landmarks = facial_landmarks.split(';')
                writer.writerow((image_full_path, bbox_xywh, facial_landmarks, expression))
        finally:
            fw.close()


if __name__ == '__main__':
    input_file = os.path.join(config.dataset.afn.csv_data, 'training.csv')
    output_file = os.path.join(config.dataset.afn.csv_data, 'train_cleaned.csv')
    clean_csv(input_file, output_file)
    input_file = os.path.join(config.dataset.afn.csv_data, 'validation.csv')
    output_file = os.path.join(config.dataset.afn.csv_data, 'val_cleaned.csv')
    clean_csv(input_file, output_file)
