#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/23 11:54
# @Author  : NYY
# @Site    : www.niuyuanyuanna.git.io
# @File    : dan_util.py
import numpy as np
import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


def e_distance(a, b):
    return np.linalg.norm(a - b)


def find_closest(matrix_list, target_matrix):
    min_distance = 999999
    closest_matrix = None
    closest_id = None
    for i, m in enumerate(matrix_list):
        if len(m) > 0:
            dist = e_distance(m, target_matrix)
            if dist < min_distance:
                closest_matrix = m
                closest_id = i
                min_distance = dist
    return closest_matrix, closest_id


def load_from_pts(filename):
    landmarks = np.genfromtxt(filename, skip_header=3, skip_footer=1)
    landmarks = landmarks - 1
    return landmarks


def save_to_pts(filename, landmarks):
    pts = landmarks + 1
    header = 'version: 1\nn_points: {}\n{{'.format(pts.shape[0])
    np.savetxt(filename, pts, delimiter=' ', header=header,
               footer='}', fmt='%.3f', comments='')


def best_fit_rect(points, meanS, box=None):
    if box is None:
        box = np.array([points[:, 0].min(), points[:, 1].min(),
                        points[:, 0].max(), points[:, 1].max()])
    boxCenter = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

    boxWidth = box[2] - box[0]
    boxHeight = box[3] - box[1]

    meanShapeWidth = meanS[:, 0].max() - meanS[:, 0].min()
    meanShapeHeight = meanS[:, 1].max() - meanS[:, 1].min()

    scaleWidth = boxWidth / meanShapeWidth
    scaleHeight = boxHeight / meanShapeHeight
    scale = (scaleWidth + scaleHeight) / 2

    S0 = meanS * scale

    S0Center = [(S0[:, 0].min() + S0[:, 0].max()) / 2,
                (S0[:, 1].min() + S0[:, 1].max()) / 2]
    S0 += boxCenter - S0Center

    return S0


def best_fit(destination, source, returnTransform=False):
    destMean = np.mean(destination, axis=0)
    srcMean = np.mean(source, axis=0)

    srcVec = (source - srcMean).flatten()
    destVec = (destination - destMean).flatten()

    a = np.dot(srcVec, destVec) / np.linalg.norm(srcVec)**2
    b = 0
    for i in range(destination.shape[0]):
        b += srcVec[2 * i] * destVec[2 * i + 1] - \
            srcVec[2 * i + 1] * destVec[2 * i]
    b = b / np.linalg.norm(srcVec)**2

    T = np.array([[a, b], [-b, a]])
    srcMean = np.dot(srcMean, T)

    if returnTransform:
        return T, destMean - srcMean
    else:
        return np.dot(srcVec.reshape((-1, 2)), T) + destMean


def mirror_shape(shape, imgShape=None):
    imgShapeTemp = np.array(imgShape)
    shape2 = mirror_shapes(shape.reshape((1, -1, 2)),
                          imgShapeTemp.reshape((1, -1)))[0]

    return shape2


def mirror_shapes(shapes, imgShapes=None):
    shapes2 = shapes.copy()

    for i in range(shapes.shape[0]):
        if imgShapes is None:
            shapes2[i, :, 0] = -shapes2[i, :, 0]
        else:
            shapes2[i, :, 0] = -shapes2[i, :, 0] + imgShapes[i][1]

        lEyeIndU = list(range(36, 40))
        lEyeIndD = [40, 41]
        rEyeIndU = list(range(42, 46))
        rEyeIndD = [46, 47]
        lBrowInd = list(range(17, 22))
        rBrowInd = list(range(22, 27))

        uMouthInd = list(range(48, 55))
        dMouthInd = list(range(55, 60))
        uInnMouthInd = list(range(60, 65))
        dInnMouthInd = list(range(65, 68))
        noseInd = list(range(31, 36))
        beardInd = list(range(17))

        lEyeU = shapes2[i, lEyeIndU].copy()
        lEyeD = shapes2[i, lEyeIndD].copy()
        rEyeU = shapes2[i, rEyeIndU].copy()
        rEyeD = shapes2[i, rEyeIndD].copy()
        lBrow = shapes2[i, lBrowInd].copy()
        rBrow = shapes2[i, rBrowInd].copy()

        uMouth = shapes2[i, uMouthInd].copy()
        dMouth = shapes2[i, dMouthInd].copy()
        uInnMouth = shapes2[i, uInnMouthInd].copy()
        dInnMouth = shapes2[i, dInnMouthInd].copy()
        nose = shapes2[i, noseInd].copy()
        beard = shapes2[i, beardInd].copy()

        lEyeIndU.reverse()
        lEyeIndD.reverse()
        rEyeIndU.reverse()
        rEyeIndD.reverse()
        lBrowInd.reverse()
        rBrowInd.reverse()

        uMouthInd.reverse()
        dMouthInd.reverse()
        uInnMouthInd.reverse()
        dInnMouthInd.reverse()
        beardInd.reverse()
        noseInd.reverse()

        shapes2[i, rEyeIndU] = lEyeU
        shapes2[i, rEyeIndD] = lEyeD
        shapes2[i, lEyeIndU] = rEyeU
        shapes2[i, lEyeIndD] = rEyeD
        shapes2[i, rBrowInd] = lBrow
        shapes2[i, lBrowInd] = rBrow

        shapes2[i, uMouthInd] = uMouth
        shapes2[i, dMouthInd] = dMouth
        shapes2[i, uInnMouthInd] = uInnMouth
        shapes2[i, dInnMouthInd] = dInnMouth
        shapes2[i, noseInd] = nose
        shapes2[i, beardInd] = beard

    return shapes2


def cyclic_learning_rate(global_step, learning_rate=0.01, max_lr=0.1, step_size=20,
                         gamma=0.99994, mode='triangular', name=None):
    if global_step is None:
        raise ValueError(
            "global_step is required for cyclic_learning_rate.")

    with ops.name_scope(name, "CyclicLearningRate",
                        [learning_rate, global_step]) as name:
        learning_rate = ops.convert_to_tensor(
            learning_rate, name="learning_rate")
        dtype = learning_rate.dtype
        global_step = math_ops.cast(global_step, dtype)
        step_size = math_ops.cast(step_size, dtype)

        def cyclic_lr():
            """Helper to recompute learning rate; most helpful in eager-mode."""
            # computing: cycle = floor( 1 + global_step / ( 2 * step_size ) )
            double_step = math_ops.multiply(2., step_size)
            global_div_double_step = math_ops.divide(global_step, double_step)
            cycle = math_ops.floor(math_ops.add(1., global_div_double_step))

            # computing: x = abs( global_step / step_size – 2 * cycle + 1 )
            double_cycle = math_ops.multiply(2., cycle)
            global_div_step = math_ops.divide(global_step, step_size)
            tmp = math_ops.subtract(global_div_step, double_cycle)
            x = math_ops.abs(math_ops.add(1., tmp))

            # computing: clr = learning_rate + ( max_lr – learning_rate ) *
            # max( 0, 1 - x )
            a1 = math_ops.maximum(0., math_ops.subtract(1., x))
            a2 = math_ops.subtract(max_lr, learning_rate)
            clr = math_ops.multiply(a1, a2)

            if mode == 'triangular2':
                clr = math_ops.divide(clr, math_ops.cast(math_ops.pow(2, math_ops.cast(
                    cycle - 1, tf.int32)), tf.float32))
            if mode == 'exp_range':
                clr = math_ops.multiply(math_ops.pow(gamma, global_step), clr)

            return math_ops.add(clr, learning_rate, name=name)

        if not context.executing_eagerly():
            cyclic_lr = cyclic_lr()

        return cyclic_lr
