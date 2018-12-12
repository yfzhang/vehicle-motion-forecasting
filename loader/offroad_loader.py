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

    def __getitem__(self, index):
        data_mat = sio.loadmat(self.data_list[index])
        feat, future_traj, past_traj = data_mat['feat'].copy(), data_mat['future_traj'], data_mat['past_traj']

        # normalize height feature locally
        feat[0] = (feat[0] - np.mean(feat[0])) / np.std(feat[0])

        # normalize other futures globally
        feat[1] = (feat[1] - self.data_normalization['variance_mean']) / self.data_normalization['variance_std']
        feat[2] = (feat[2] - self.data_normalization['red_mean']) / self.data_normalization['red_std']
        feat[3] = (feat[3] - self.data_normalization['green_mean']) / self.data_normalization['green_std']
        feat[4] = (feat[4] - self.data_normalization['blue_mean']) / self.data_normalization['blue_std']

        if self.pre_train:
            target = data_mat['feat'][1].copy()  # copy the variance layer first
            target[target < 0.5] = 0.0
            target[target >= 0.5] = -1.0
            return feat, target


        traj = np.vstack((past_traj[:,:2].copy(), future_traj[:, :2].copy()))
        traj = self.correct_connection(traj)

        traj = self.auto_pad(traj)
        return feat, traj

    def __len__(self):
        return len(self.data_list)

    def correct_connection(self, traj):
        i = 0
        while i < traj.shape[0]-1:
            [dx, dy] = traj[i+1] - traj[i]
            copy = traj[i+1].copy()
            while (abs(dx)+abs(dy)) > 1:
                if abs(dx) != 0:
                    traj = np.insert(traj, i+1, copy - [np.sign(dx), 0], axis=0)
                else:
                    traj = np.insert(traj, i+1, copy - [0, np.sign(dy)], axis=0)
                [dx, dy] = copy - traj[i+1]
                i+=1
            i+=1
        i = 0
        while i < traj.shape[0]-1:
            while (traj[i]==traj[i+1]).all():
                traj = np.delete(traj, i, axis=0)
            i += 1
        return traj

    def auto_pad(self, traj):
        """
        add padding (NAN) to traj to keep traj length fixed.
        traj shape needs to be fixed in order to use batch sampling
        :param traj: numpy array. (traj_len, 2)
        :return:
        """
        if traj.shape[0] >= self.grid_size*2:
            raise ValueError('traj length {} must be less than grid_size*2 {}'.format(traj.shape[0], self.grid_size))
        pad_len = self.grid_size*2 - traj.shape[0]
        pad_array = np.full((pad_len, 2), np.nan)
        output = np.vstack((traj, pad_array))
        return output
