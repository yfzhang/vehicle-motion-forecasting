#!/usr/bin/env python
# -*- coding: utf-8 -*-

from loader.offroad_loader import *
import unittest
import visdom
import numpy as np
from maxent_nonlinear_offroad import overlay_traj_to_map

np.set_printoptions(threshold=np.inf)
import logging
import torch
from torch.utils.data import DataLoader


class TestOffroadLoader(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vis = visdom.Visdom()

    def test_dataset(self):
        grid_size = 60
        n_traj = 20
        loader = OffroadLoader(grid_size=grid_size)
        for idx in range(3):
            feat, traj = loader[idx]  # the index does not matter. random sample inside loader
            if check_connectivity(traj) is not True:
                print("not connected")

    def test_loader(self):
        grid_size = 60
        n_traj = 10
        loader = OffroadLoader(grid_size=grid_size)
        loader = DataLoader(loader, num_workers=1, batch_size=1, shuffle=True)
        # for idx, (feat, traj) in enumerate(loader):
        #     if idx > 3:
        #         break
        #     feat = torch.squeeze(feat,dim=0)
        #     self.vis.heatmap(X=feat[0], opts=dict(colormap='Electric', title='height_max. idx {}'.format(idx)))
        #     self.vis.heatmap(X=feat[1], opts=dict(colormap='Electric', title='height_var. idx {}'.format(idx)))
        #     self.vis.heatmap(X=feat[2], opts=dict(colormap='Electric', title='red. idx {}'.format(idx)))
        #     self.vis.heatmap(X=feat[3], opts=dict(colormap='Electric', title='green. idx {}'.format(idx)))
        #     self.vis.heatmap(X=feat[4], opts=dict(colormap='Electric', title='blue. idx {}'.format(idx)))
        #     traj = torch.squeeze(traj,dim=0)
        #     if check_connectivity(traj) is not True:
        #         print("not connected")

def check_connectivity(traj):
    for i in range(traj.shape[0]-1):
        [dx, dy] = traj[i+1]-traj[i]
        if (abs(dx)+abs(dy)) != 1:
            print(traj[i])
            print(traj[i+1])
            return False
    return True


if __name__ == '__main__':
    logging.basicConfig(filename='test_offroad_loader_v2.log', format='%(levelname)s. %(asctime)s. %(message)s',
                        level=logging.DEBUG)
    unittest.main()
