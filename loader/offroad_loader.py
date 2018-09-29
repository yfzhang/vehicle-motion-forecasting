#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from torch.utils.data import Dataset
import os
import scipy.io as sio
from skimage.transform import resize
from skimage.transform import rotate
import numpy as np
from scipy import optimize

np.set_printoptions(threshold=np.inf, suppress=True)
import random
import copy
import math


class OffroadLoader(Dataset):
    def __init__(self, grid_size, train=True, demo=None, datadir='/data/datasets/yanfu', pre_train=False, tangent=False,
                 more_kinematic=None):
        assert grid_size % 2 == 0, "grid size must be even number"
        self.grid_size = grid_size
        if train:
            self.data_dir = datadir + '/irl_data/train_data'
        else:
            self.data_dir = datadir + '/irl_data/test_data'

        if demo is not None:
            self.data_dir = datadir + '/irl_data/' + demo

        items = os.listdir(self.data_dir)
        self.data_list = []
        for item in items:
            self.data_list.append(self.data_dir + '/' + item)

        self.data_normalization = sio.loadmat(datadir + '/irl_data/train-data-mean-std.mat')
        self.pre_train = pre_train
        self.tangent = tangent
        self.more_kinematic = more_kinematic

        # kinematic related feature
        self.center_idx = self.grid_size / 2
        self.delta_x_layer = np.zeros((self.grid_size, self.grid_size), dtype=np.float)
        self.delta_y_layer = self.delta_x_layer.copy()
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                self.delta_x_layer[x, y] = x - self.center_idx
                self.delta_y_layer[x, y] = y - self.center_idx
                # self.delta_x_layer[x, y] = x
                # self.delta_y_layer[x, y] = y

    def __getitem__(self, index):
        data_mat = sio.loadmat(self.data_list[index])
        feat, future_traj, past_traj = data_mat['feat'].copy(), data_mat['future_traj'], data_mat['past_traj']
        # normalize height feature locally
        feat[0] = (feat[0] - np.mean(feat[0])) / np.std(feat[0])
        # normalize the rest futures globally
        feat[1] = (feat[1] - self.data_normalization['variance_mean']) / self.data_normalization['variance_std']
        feat[2] = (feat[2] - self.data_normalization['red_mean']) / self.data_normalization['red_std']
        feat[3] = (feat[3] - self.data_normalization['green_mean']) / self.data_normalization['green_std']
        feat[4] = (feat[4] - self.data_normalization['blue_mean']) / self.data_normalization['blue_std']

        if self.more_kinematic is not None:
            if random.random() < self.more_kinematic:
                feat[0] = random.gauss(-0.5, 0.01)
                feat[1] = random.gauss(-0.5, 0.01)
                feat[2] = 2.4697
                feat[3] = 2.3534
                feat[4] = 2.7742

        # kinematic features
        x, y = past_traj[:, 0], past_traj[:, 1]
        xc, yc, r, _ = self.leastsq_circle(x, y)
        x1, y1, x2, y2 = x[0], y[0], x[-1], y[-1]
        curve_sign = self.calc_sign(x1, y1, x2, y2, xc, yc)
        normalized_curvature = 1.0 / r * curve_sign * 10.0  # 10.0 is empirically choosen by observing the histogram
        feat = np.vstack((feat, np.full((1, self.grid_size, self.grid_size), normalized_curvature, dtype=np.float)))

        normalization = 0.5 * self.grid_size
        if self.tangent:
            # direction as circle tangent, speed as past length
            y_end, x_end = past_traj[-1, 0], past_traj[-1, 1]
            x_sign = np.sign(x_end - xc)
            y_sign = np.sign(y_end - yc)

            speed = (np.abs(past_traj[-1, 0] - past_traj[0, 0]) + np.abs(past_traj[-1, 1] - past_traj[0, 1])) / normalization
            a = np.abs(y_end - yc) / np.abs(x_end - xc)  # vx_abs = vy_abs * a
            vy_abs = speed / (a + 1)
            vx_abs = a * vy_abs

            vx = y_sign * curve_sign * vx_abs * -1
            vy = x_sign * curve_sign * vy_abs * -1
            # print('vx {}, vy {}, r {}, index {}'.format(vx, vy, r, index))
        else:
            vx = (past_traj[-1, 0] - past_traj[0, 0]) / normalization
            vy = (past_traj[-1, 1] - past_traj[0, 1]) / normalization

        feat = np.vstack((feat, np.full((1, self.grid_size, self.grid_size), vx, dtype=np.float)))
        feat = np.vstack((feat, np.full((1, self.grid_size, self.grid_size), vy, dtype=np.float)))
        feat = np.vstack((feat, np.expand_dims(self.delta_x_layer.copy() / normalization, axis=0)))
        feat = np.vstack((feat, np.expand_dims(self.delta_y_layer.copy() / normalization, axis=0)))

        if self.pre_train:
            target = data_mat['feat'][1].copy()  # copy the variance layer first
            target[target < 0.5] = 0.0
            target[target >= 0.5] = -1.0
            return feat, target

        future_traj = self.auto_pad(future_traj[:, :2])
        past_traj = self.auto_pad(past_traj[:, :2])

        return feat, past_traj, future_traj

    def __len__(self):
        return len(self.data_list)

    def auto_pad(self, traj):
        """
        add padding (NAN) to traj to keep traj length fixed.
        traj shape needs to be fixed in order to use batch sampling
        :param traj: numpy array. (traj_len, 2)
        :return:
        """
        fixed_len = self.grid_size
        if traj.shape[0] >= self.grid_size:
            raise ValueError('traj length {} must be less than grid_size {}'.format(traj.shape[0], self.grid_size))
        pad_len = self.grid_size - traj.shape[0]
        pad_array = np.full((pad_len, 2), np.nan)
        output = np.vstack((traj, pad_array))
        return output

    @staticmethod
    def leastsq_circle(x, y):
        def calc_radius(x, y, xc, yc):
            """ calculate the distance of each 2D points from the center (xc, yc) """
            return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

        def f(c, x, y):
            """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
            ri = calc_radius(x, y, *c)
            return ri - ri.mean()

        # coordinates of the barycenter
        x_m = np.mean(x)
        y_m = np.mean(y)
        center_estimate = x_m, y_m
        center, ier = optimize.leastsq(f, center_estimate, args=(x, y))
        xc, yc = center
        ri = calc_radius(x, y, *center)
        r = ri.mean()
        residu = np.sum((ri - r) ** 2)
        return xc, yc, r, residu

    @staticmethod
    def calc_sign(x1, y1, x2, y2, xc, yc):
        theta1 = math.atan2(y1 - yc, x1 - xc)
        theta2 = math.atan2(y2 - yc, x2 - xc)
        return np.sign(theta1 - theta2)
