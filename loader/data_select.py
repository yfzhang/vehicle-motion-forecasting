#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from torch.utils.data import Dataset
import os
import scipy.io as sio
from skimage.transform import resize
from skimage.transform import rotate
import numpy as np

np.set_printoptions(threshold=np.inf, suppress=True)
import random
import copy
import math
from scipy.io import savemat
import visdom

"""
for the whole point cloud map,
feat[1], height var, mean is 3.8140, std is 5.9498
feat[2], red mean is 101.5115, std is 42.8356
feat[3], green mean is 115.9497, std is 37.5907
feat[4], blue mean is 81.2158, std is 37.5544
"""


def overlay_traj_to_map(traj_idx, feat_map, value):
    overlay_map = copy.deepcopy(feat_map)
    logging.info('overlay_traj_to_map.\n traj_idx {}'.format(traj_idx))
    for i, p in enumerate(traj_idx):
        if math.fabs(curvature[i]) > 0.5:
            overlay_map[int(p[0]),int(p[1])] = value + 20
        else:
            overlay_map[int(p[0]), int(p[1])] = value
    return overlay_map


def curvature_thresholding(traj):
    distance_thresh = 4
    curvature = np.zeros(traj.shape[0], dtype=np.float)
    for i in range(traj.shape[0]):
        plus_i = 1
        plus_distance = 0
        while plus_distance < distance_thresh:
            if i + plus_i >= traj.shape[0]:
                break
            plus_distance = np.sum(np.abs(traj[i + plus_i, :2] - traj[i, :2]))
            plus_i += 1

        minus_i = -1
        minus_distance = 0
        while minus_distance < distance_thresh:
            if i + minus_i <= 0:
                break
            minus_distance = np.sum(np.abs(traj[i] - traj[i + minus_i]))
            minus_i -= 1

        if minus_distance >= distance_thresh and plus_distance >= distance_thresh:
            plus_heading = math.atan2(traj[i + plus_i, 0] - traj[i, 0], traj[i + plus_i, 1] - traj[i, 1])
            minus_heading = math.atan2(traj[i, 0] - traj[i + minus_i, 0], traj[i, 1] - traj[i + minus_i, 1])
            curvature[i] = plus_heading - minus_heading
            print('curvature {} at i {}'.format(curvature[i], i))

    return curvature



vis = visdom.Visdom(env='main')

traj = sio.loadmat('/data/datasets/yanfu/irl_data/idx_traj.mat')['idx_traj']
traj_idx = traj[~np.isnan(traj).any(axis=1)].copy()
curvature = curvature_thresholding(traj_idx)

feat_map = sio.loadmat('/data/datasets/yanfu/irl_data/gascola_grid_map_full.mat')['grid_map']
variance_map = feat_map[1]
variance_map[variance_map > 50] = 50.0
np.nan_to_num(variance_map, copy=False)

map = overlay_traj_to_map(traj_idx, variance_map, 50.0)
vis.heatmap(X=map, opts=dict(colormap='Electric', title='map'))
print('finish')
