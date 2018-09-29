#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from torch.utils.data import Dataset
import os
import os.path as path
import scipy.io as sio
from skimage.transform import resize
from skimage.transform import rotate
import numpy as np

np.set_printoptions(threshold=np.inf, suppress=True)
import random
from torch.utils.data import DataLoader
import copy
import math


class KinematicLoader(Dataset):
    def __init__(self, grid_size, n_traj):
        self.grid_size = grid_size
        self.n_traj = n_traj
        self.center_idx = int(self.grid_size / 2)

        self.delta_x_layer = np.zeros((self.grid_size, self.grid_size), dtype=np.float)
        self.delta_y_layer = self.delta_x_layer.copy()
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                self.delta_x_layer[x, y] = x - self.center_idx
                self.delta_y_layer[x, y] = y - self.center_idx

    def __getitem__(self, index):
        while True:
            goal_idx = np.random.randint(low=0, high=self.grid_size, size=2)

            # use 0.5 * grid_size to normalize
            normalization = 0.5 * self.grid_size
            vx = (goal_idx[0] - self.center_idx) / normalization
            vy = (goal_idx[1] - self.center_idx) / normalization

            feat = np.zeros((4, self.grid_size, self.grid_size))
            feat[0, :, :].fill(vx)
            feat[1, :, :].fill(vy)

            # use 0.5 * grid size to normalize
            feat[2, :, :] = self.delta_x_layer.copy() / normalization
            feat[3, :, :] = self.delta_y_layer.copy() / normalization

            traj = []
            current_pos = [self.center_idx, self.center_idx]
            while (goal_idx != current_pos).all():
                traj.append(current_pos.copy())
                if abs(goal_idx[0] - current_pos[0]) > abs(goal_idx[1] - current_pos[1]):
                    if goal_idx[0] > current_pos[0]:
                        current_pos[0] += 1
                        continue
                    else:
                        current_pos[0] -= 1
                        continue
                else:
                    if goal_idx[1] > current_pos[1]:
                        current_pos[1] += 1
                        continue
                    else:
                        current_pos[1] -= 1
                        continue

            traj.append(current_pos.copy())
            if len(traj) < 5:
                print('getitem. traj too short. resample')
                continue
            traj = np.asanyarray(traj)

            return feat, traj

    def __len__(self):
        return self.n_traj
