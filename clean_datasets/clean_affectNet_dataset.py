#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/31 11:47
# @Author  : NYY
# @Site    : www.niuyuanyuanna.git.io
# @File    : clean_affectNet_dataset.py
import csv
import os
from PIL import Image
import cv2

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
                    fe.write(image_full_path + '\t' + bbox_xywh + '\t' + facial_landmarks + '\t' + expression + '\n')


def clean_txt(input_filepath, output_filepath):
    with open(input_filepath, 'r') as fin:
        count = 0
        with open(output_filepath, 'w') as fout:
            for line_o in fin:
                count += 1
                line = line_o.split('\t')
                img_path = line[0]
                try:
                    img = cv2.imread(img_path)
                    w, h, c = img.shape
                    print('deal with the %d th img, w=%d, h=%d' % (count, w, h))
                    fout.write(line_o)
                except Exception as e:
                    print(Exception, ":", e)
                    continue
    print('done')


# ====================clean txt file ==============

def clean_train_cl(input_filepath, output_filepath):
    with open(input_filepath, 'rb') as fin:
        count = 0
        with open(output_filepath, 'a') as fout:
            for line in fin:
                count += 1
                try:
                    line_origin = line.decode()
                    line = line_origin.split('\t')
                    print('deal with the %d th img' % count)
                    fout.write(line_origin)
                except Exception as e:
                    print(Exception, ":", e)
                    continue
    print('done')


def delete_ret(input_filepath, output_filepath):
        fin = open(input_filepath, 'r')
        fnew = open(output_filepath, 'w')  # 将结果存入新的文本中
        for line in fin.readlines():  # 对每一行先删除空格，\n等无用的字符，再检查此行是否长度为0
            data = line.strip()
            if len(data) != 0:
                fnew.write(data)
                fnew.write('\n')
        fin.close()
        fnew.close()


def check_each_line(input_filepath, output_filepath):
    fin = open(input_filepath, 'r')
    fout = open(output_filepath, 'w')
    count = 0
    for line in fin:
        count += 1
        try:
            line_list = line.split('\t')
            img_path = line_list[0]
            bbox = line_list[1]
            landmarks = line_list[2].split(';')
            if not len(landmarks) == 136:
                print('line %d is invalid in landmarks' % count)
                continue

            emotion = int(line_list[-1])
            print('deal with line %d done' % count)
            fout.write(line)
        except Exception as e:
            print(Exception, ":", e)
            continue
    fin.close()
    fout.close()
    print('done')


if __name__ == '__main__':
    # input_file = os.path.join(config.dataset.afn.csv_data, 'training.csv')
    # output_file = os.path.join(config.dataset.afn.csv_data, 'train_c.txt')
    # clean_csv(input_file, output_file)
    # input_file = os.path.join(config.dataset.afn.csv_data, 'validation.csv')
    # output_file = os.path.join(config.dataset.afn.csv_data, 'val_c.txt')
    # clean_csv(input_file, output_file)

    # out_f = os.path.join(config.dataset.afn.csv_data, 'train_cl.txt')
    input_file = os.path.join(config.dataset.afn.csv_data, 'val_c.txt')
    out_val = os.path.join(config.dataset.afn.csv_data, 'val_cl.txt')
    clean_txt(input_file, out_val)
    # out2 = os.path.join(config.dataset.afn.csv_data, 'train_c2.txt')
    # out3 = os.path.join(config.dataset.afn.csv_data, 'train_c3.txt')
    # out4 = os.path.join(config.dataset.afn.csv_data, 'train_c4.txt')
    # clean_train_cl(out_f, out2)
    # delete_ret(out2, out3)
    # check_each_line(out_f, out4)
