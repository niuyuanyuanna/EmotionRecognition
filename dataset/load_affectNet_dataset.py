#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/26 16:08
# @Author  : NYY
# @Site    : www.niuyuanyuanna.git.io
# @File    : load_affectNet_dataset.py
import os
import glob
import csv
from six.moves import cPickle as pickle
from scipy import ndimage
from matplotlib import pyplot as plt

from util.dan_util import *
from config.configs import config


class AffetNet_Server(object):

    def __init__(self, img_size=[112, 112], frame_fraction=0.25,
                 initialization='box', color=False):
        self.orig_landmark = []
        self.filenames = []
        self.mirrors = []
        self.mean_shape = np.array([])

        self.img_mean = np.array([])
        self.img_std = np.array([])

        self.perturbations = []

        self.img_size = img_size
        self.fram_fraction = frame_fraction
        self.initialization = initialization
        self.color = color

        self.boundingbox = []
        self.emotions = []

    @staticmethod
    def load(filename):
        imageServer = AffetNet_Server()
        arrays = np.load(filename)
        imageServer.__dict__.update(arrays)
        if len(imageServer.imgs.shape) == 3:
            imageServer.imgs = imageServer.imgs[:, :, :, np.newaxis]
        return imageServer

    def save(self, dataset_path, filename=None):
        if filename is None:
            filename = 'dataset_nimgs={0}_perturbations={1}_size={2}'.format(
                len(self.imgs), list(self.perturbations), self.img_size)
            if self.color:
                filename += '_color={}'.format(self.color)
            filename += '.npz'
        arrays = {key: value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(key)}
        dataset_path = os.path.join(dataset_path, filename)
        np.savez(dataset_path, **arrays)

    def prepare_data(self, csv_file, mean_shape, start_id, n_images, mirror_flag):
        filenames = []
        landmarks = []
        boundingboxs = []
        emotions = []

        with open(csv_file, 'rb') as f:
            reader = csv.reader(f)
            headers = next(reader)
            for row in reader:
                subDirectory_filePath = row[0]
                image_full_path = os.path.join(config.dataset.afn.image_path, subDirectory_filePath)
                bbox_xywh = row[1:5]
                facial_landmarks = row[5]
                facial_landmarks = facial_landmarks.split(';')
                expression = row[6]

                filenames.append(image_full_path)
                landmarks.append(facial_landmarks)
                boundingboxs.append(bbox_xywh)
                emotions.append(expression)

        filenames = filenames[start_id: start_id + n_images]
        landmarks = landmarks[start_id: start_id + n_images]
        boundingboxs = boundingboxs[start_id: start_id + n_images]
        emotions = emotions[start_id: start_id + n_images]

        mirror_list = [False for i in range(n_images)]
        if mirror_flag:
            mirror_list = mirror_list + [True for i in range(n_images)]
            filenames = np.concatenate((filenames, filenames))
            landmarks = np.vstack((landmarks, landmarks))
            boundingboxs = np.vstack((boundingboxs, boundingboxs))
            emotions = np.vstack((emotions, emotions))

        self.orig_landmark = landmarks
        self.filenames = filenames
        self.mirrors = mirror_list
        self.mean_shape = mean_shape
        self.boundingbox = boundingboxs
        self.emotions = emotions

    def load_images(self):
        self.imgs = []
        self.init_landmarks = []
        self. gt_landmarks = []
        self.gt_emotions = []
        for i in range(len(self.filenames)):
            img = ndimage.imread(self.filenames[i])
            if self.color:
                if len(img.shape) == 2:
                    img = np.dstack((img, img, img))
            else:
                if len(img.shape) > 2:
                    img = np.mean(img, axis=2)
            img = img.astype(np.uint8)

            if self.mirrors[i]:
                self.orig_landmark[i] = mirror_shape(self.orig_landmark[i], img.shape)
                img = np.fliplr(img)
            if not self.color:
                img = img[np.newaxis]
            gt = self.orig_landmark[i]
            if self.initialization == 'rect':
                best_fit_region = best_fit_rect(gt, self.mean_shape)
            elif self.initialization == 'similarity':
                best_fit_region = best_fit(gt, self.mean_shape)
            elif self.initialization == 'box':
                best_fit_region = best_fit_rect(gt, self.mean_shape, box=self.boundingbox[i])
            gt_emotion = self.emotions[i]
            self.gt_emotions.append(gt_emotion)
            self.imgs.append(img)
            self.init_landmarks.append(best_fit_region)
            self.gt_landmarks.append(gt)
        self.init_landmarks = np.array(self.init_landmarks)
        self.gt_landmarks = np.array(self.gt_landmarks)

    def generate_perturbation(self, n_perturbation, perturbations):
        self.perturbations = perturbations
        mean_shap_size = max(self.mean_shape.max(axis=0) - self.mean_shape.min(axis=0))
        dest_shape_size = min(self.img_size) * (1 - 2 * self.fram_fraction)
        scaled_mean_shape = self.mean_shape * dest_shape_size / mean_shap_size

        new_img = []
        new_gt_landmarks = []
        new_ini_landmarks = []

        translation_mult_X, translation_mult_Y, rotation_std, scale_std = perturbations
        rotation_std_rad = rotation_std * np.pi / 180
        translation_std_X = translation_mult_X * (scaled_mean_shape[:, 0].max() - scaled_mean_shape[:, 0].min())
        translation_std_Y = translation_mult_Y * (scaled_mean_shape[:, 1].max() - scaled_mean_shape[:, 1].min())
        print('creating perturbations of %d shapes' % self.gt_landmarks.shape[0])

        for i in range(self.init_landmarks.shape[0]):
            for j in range(n_perturbation):
                temp_init = self.init_landmarks[i].copy()
                angle = np.random.normal(0, rotation_std_rad)
                offset = [np.random.normal(0, translation_std_X), np.random.normal(0, translation_std_Y)]
                scaling = np.random.normal(1, scale_std)

                R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

                temp_init = temp_init + offset
                temp_init = (temp_init - temp_init.mean(axis=0)) * scaling + temp_init.mean(axis=0)
                temp_init = np.dot(R, (temp_init - temp_init.mean(axis=0)).T).T + temp_init.mean(axis=0)
                temp_img, temp_init, temp_gt = self.crop_resize_rotate(self.imgs[i], temp_init, self.gt_landmarks[i])

                new_img.append(temp_img.transpose(1, 2, 0))
                new_ini_landmarks.append(temp_init)
                new_gt_landmarks.append(temp_gt)
        self.imgs = np.array(new_img)
        self.init_landmarks = np.array(new_ini_landmarks)
        self.gt_landmarks = np.array(new_gt_landmarks)

    def crop_resize_rotate(self, img, init_shape, gt):
        mean_shape_size = max(self.mean_shape.max(axis=0) - self.mean_shape.min(axis=0))
        dest_shape_size = min(self.img_size) * (1 - 2 * self.fram_fraction)
        scaled_mean_shap = self.mean_shape * dest_shape_size / mean_shape_size
        dest_shape = scaled_mean_shap.copy() - scaled_mean_shap.mean(axis=0)
        offset = np.array(self.img_size[::-1]) / 2
        dest_shape += offset
        A, t = best_fit(dest_shape, init_shape, True)
        A2 = np.linalg.inv(A)
        t2 = np.dot(-t, A2)

        out_img = np.zeros((img.shape[0], self.img_size[0], self.img_size[1]), dtype=img.dtype)
        for i in range(img.shape[0]):
            out_img[i] = ndimage.interpolation.affine_transform(img[i], A2, t2[[1, 0]], output_shape=self.img_size)

        init_shape = np.dot(init_shape, A) + t
        gt = np.dot(gt, A) + t
        return out_img, init_shape, gt

    def crop_resize_rotate_all(self):
        new_imgs = []
        new_gt_landmarks = []
        new_init_landmarks = []
        for i in range(self.init_landmarks.shape[0]):
            temp_img, temp_init, temp_gt = self.crop_resize_rotate(self.imgs[i], self.init_landmarks[i], self.gt_landmarks[i])
            new_imgs.append(temp_img.transpose((1, 2, 0)))
            new_init_landmarks.append(temp_init)
            new_gt_landmarks.append(temp_gt)
        self.imgs = np.array(new_imgs)
        self.init_landmarks = np.array(new_init_landmarks)
        self.gt_landmarks = np.array(new_gt_landmarks)

    def normalize_imgs(self, image_server=None):
        self.imgs = self.imgs.astype(np.float32)
        if image_server is None:
            self.img_mean = np.mean(self.imgs, axis=0)
            self.img_std = np.std(self.imgs, axis=0)
        else:
            self.img_mean = image_server.img_mean
            self.img_std = image_server.img_std
        self.imgs = self.imgs - self.img_mean
        self.imgs = self.imgs / self.img_std

        mean_img = self.img_mean - self.img_mean.min()
        mean_img = 255 * mean_img / mean_img.max()
        mean_img = mean_img.astype(np.uint8)

        std_img = self.img_std - self.img_std.min()
        std_img = 255 * std_img / std_img.max()
        std_img = std_img.astype(np.uint8)

        plt.figure(num='sample')






















