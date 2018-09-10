# -*- coding: UTF-8 -*-
# 预处理CK+数据集，提取图片数据到对应label文件夹下
# 便于迁移训练load_data
import os
import shutil
from config.configs import config


def process_dataset():
    image_base_path = config.ck_origin_img_path
    label_base_path = config.cd_label_data_path
    save_base_path = config.ck_enhanced_img_path

    n = 0
    basefile_dir = os.listdir(image_base_path)
    label_list = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

    for root, dirs, files in os.walk(label_base_path):
        for i in range(len(files)):
            if files[i][-3:] == 'txt':
                arr = files[i].split("_")
                image_path = os.path.join(image_base_path, arr[0], arr[1])
                save_path = os.path.join(save_base_path, arr[0])  # 图片保存路径

                label_file_name = os.path.join(root, files[i])
                with open(label_file_name) as lf:
                    line = lf.readline()
                    label_id = line.split('.')
                    label_name = label_list[int(label_id[0][-1]) - 1]
                    n += 1

                    for root1, dir1, images in os.walk(image_path):
                        if images[i][-3:] == 'png':
                            for im in range(len(images)):
                                if not os.path.exists(os.path.join(save_base_path, label_name)):
                                    os.makedirs(os.path.join(save_base_path, label_name))
                                oragin_img_path = os.path.join(image_path, images[im])
                                image_name = label_name + '_' + str(n) + '.png'
                                save_img_path = os.path.join(save_base_path, label_name, image_name)
                                shutil.copy(oragin_img_path, save_img_path)
                                print(os.path.join(save_base_path, label_name, label_name+'_'+ str(n)))




