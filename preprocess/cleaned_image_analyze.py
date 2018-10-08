#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/18 17:48
# @Author  : NYY
# @Site    : www.niuyuanyuanna.git.io
# @File    : cleaned_image_analyze.py
import json
import os
import shutil

from config.configs import config
from dataset.load_fer2013_dataset import load_all_imagePath_label_list


def get_error_images(file_path):
    error_image_list = []
    error_image_label = []
    with open(file_path, 'rb') as fid:
        for line in fid:
            line = line.decode()
            line = line.strip().split('\t')
            image_name = line[0]
            image_label = line[1]
            error_image_list.append(image_name)
            error_image_label.append(image_label)
    return error_image_list, error_image_label


def get_label_dict(image_label_list, label_list):
    label_class_dict = {}
    for label_name in label_list:
        label_class_dict[label_name] = 0
    for image_label in image_label_list:
        label_class_dict[image_label] += 1
    return label_class_dict


def get_normal_images(file_path):
    normal_image_list = []
    normal_image_label = []
    predict_image_label = []
    predict_image_prob = []
    predict_json_scores = []
    with open(file_path, 'rb') as fid:
        for line in fid:
            line = line.decode()
            line = line.strip().split('\t')
            image_name = line[0]
            image_gt_label = line[1]
            image_pre_label = line[2]
            image_pre_prob = line[3]
            image_pre_scores = line[4]
            image_pre_scores = json.loads(image_pre_scores)

            normal_image_list.append(image_name)
            normal_image_label.append(image_gt_label)
            predict_image_label.append(image_pre_label)
            predict_image_prob.append(image_pre_prob)
            predict_json_scores.append(image_pre_scores)
    return normal_image_list, normal_image_label, predict_image_label, predict_image_prob, predict_json_scores


def get_diff_list(origin_list, error_list, norm_list):
    clean_list = error_list + norm_list
    diff_list = [i for i in origin_list if i not in clean_list]
    diff_label = []
    for image_path in diff_list:
        image_name = os.path.basename(image_path)
        line = image_name.split('_')
        image_label = line[0]
        diff_label.append(image_label)
    return diff_list, diff_label


def write_list_to_file(diff_list, save_path):
    save_file = os.path.join(save_path, 'diff_list.txt')
    with open(save_file, 'w') as fid:
        for name in diff_list:
            fid.write(name + '\n')


def creat_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return


def copy_image_file(image_list, image_label, save_dir):
    creat_dir(save_dir)
    for i in range(len(image_list)):
        image_name = os.path.basename(image_list[i])
        image_ture_label = image_name.split('_')[0]
        oragin_class_path = os.path.join(config.dataset.fer2013.rebuild_image_from_csv, image_ture_label)
        origin_image_path = os.path.join(oragin_class_path, image_name)

        new_class_path = os.path.join(save_dir, image_ture_label)
        # new_class_path = os.path.join(save_dir, image_label[i])
        creat_dir(new_class_path)
        new_image_path = os.path.join(new_class_path, image_name)

        shutil.copyfile(origin_image_path, new_image_path)
    print('copy all %d images' % len(image_list))
    return


def convert_predict_label(predict_image_label, label_list, plabel_list):
    new_image_label_list = []
    for plabel in predict_image_label:
        if plabel == plabel_list[-1]:
            new_image_label_list.append(plabel)
        for i in range(0, len(plabel_list) - 1):
            if plabel == plabel_list[i]:
                new_image_label_list.append(label_list[i])
    return new_image_label_list


def split_gt_label(new_predict_label_list, normal_image_label, normal_image_list):
    right_classified_image_path = []
    right_classified_image_label = []
    wrong_classified_image_path = []
    wrong_classified_image_label = []
    for i in range(len(new_predict_label_list)):
        if new_predict_label_list[i] == normal_image_label[i]:
            right_classified_image_path.append(normal_image_list[i])
            right_classified_image_label.append(normal_image_label[i])
        else:
            wrong_classified_image_path.append(normal_image_list[i])
            wrong_classified_image_label.append(new_predict_label_list[i])
    return right_classified_image_path, right_classified_image_label, wrong_classified_image_path, wrong_classified_image_label


def handle_recognized_images(label_list, plabel_list):
    normal_file_path = os.path.join(config.dataset.fer2013.cleaned_face_imgs_path, 'norm_image.txt')
    normal_image_list, normal_image_label, predict_image_label, \
    predict_image_prob, predict_json_scores = get_normal_images(normal_file_path)
    norm_gt_label_dict = get_label_dict(normal_image_label, label_list)
    print(norm_gt_label_dict)

    normal_image_file_path = os.path.join(config.dataset.fer2013.cleaned_face_imgs_path, 'normal')
    new_predict_label_list = convert_predict_label(predict_image_label, label_list, plabel_list)
    right_classified_image_path, right_classified_image_label, \
    wrong_classified_image_path, wrong_classified_image_label = split_gt_label(new_predict_label_list,
                                                                               normal_image_label,
                                                                               normal_image_list)
    normal_right_image_file_path = os.path.join(normal_image_file_path, 'right')
    copy_image_file(right_classified_image_path, right_classified_image_label, normal_right_image_file_path)
    # normal_wrong_image_file_path = os.path.join(normal_image_file_path, 'wrong')
    # copy_image_file(wrong_classified_image_path, wrong_classified_image_label, normal_wrong_image_file_path)
    normal_wrong_gt_image_file_path = os.path.join(normal_image_file_path, 'wrong_for_gt')
    copy_image_file(wrong_classified_image_path, wrong_classified_image_label, normal_wrong_gt_image_file_path)

    return normal_image_list


def handel_total_images(label_list):
    total_file_path = 'E:/liuyuan/DataCenter/DatasetFER2013/fer2013\\padding_images'
    tatal_imagePath_list, total_label_list = load_all_imagePath_label_list(total_file_path)
    total_label_dict = get_label_dict(total_label_list, label_list)
    print(total_label_dict)
    return tatal_imagePath_list


def handel_unrecognized_images(label_list):
    error_file_path = os.path.join(config.dataset.fer2013.cleaned_face_imgs_path, 'error_image.txt')
    error_image_list, error_image_label = get_error_images(error_file_path)
    error_label_dict = get_label_dict(error_image_label, label_list)
    print(error_label_dict)
    error_image_new_path = os.path.join(config.dataset.fer2013.cleaned_face_imgs_path, 'unrecognized')
    copy_image_file(error_image_list, error_image_label, error_image_new_path)
    return error_image_list


def handel_diff_txt(tatal_imagePath_list, error_image_list, normal_image_list):
    diff_list, diff_label = get_diff_list(tatal_imagePath_list, error_image_list, normal_image_list)
    diff_label_dict = get_label_dict(diff_label, label_list)
    print(diff_label_dict)

    diff_file_path = config.dataset.fer2013.cleaned_face_imgs_path
    write_list_to_file(diff_list, diff_file_path)
    return


if __name__ == '__main__':
    label_list = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    plabel_list = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise', 'contempt']

    # tatal_imagePath_list = handel_total_images(label_list)
    # error_image_list = handel_unrecognized_images(label_list)
    normal_image_list = handle_recognized_images(label_list, plabel_list)
    # handel_diff_txt(tatal_imagePath_list, error_image_list, normal_image_list)
