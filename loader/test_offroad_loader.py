#!/usr/bin/env python
# -*- coding: utf-8 -*-

from loader.offroad_loader import *
import unittest
import visdom
import numpy as np

np.set_printoptions(threshold=np.inf)
import logging
import torch


def overlay_traj_to_map(traj_idx, feat_map, value):
    overlay_map = copy.deepcopy(feat_map)
    logging.info('overlay_traj_to_map.\n traj_idx {}'.format(traj_idx))
    for idx in traj_idx:
        overlay_map[int(idx[0]), int(idx[1])] = value
    return overlay_map


class TestOffroadLoader(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vis = visdom.Visdom()

    def test_dataset(self):
        grid_size = 60
        n_traj = 20
        loader = OffroadLoader(grid_size=grid_size, n_traj=n_traj)
        for idx in range(50):
            feat, past_traj, future_traj = loader[0]  # the index does not matter. random sample inside loader
            # self.vis.heatmap(X=feat[0], opts=dict(colormap='Electric', title='height_max. idx {}'.format(idx)))
            # self.vis.heatmap(X=feat[1], opts=dict(colormap='Electric', title='height_var. idx {}'.format(idx)))
            # overlay_map = overlay_traj_to_map(past_traj, feat[0], 3)
            # overlay_map = overlay_traj_to_map(future_traj, overlay_map, 5)
            # self.vis.heatmap(X=overlay_map, opts=dict(colormap='Electric', title='overlayed. idx {}'.format(idx)))
            # self.vis.heatmap(X=feat[5], opts=dict(colormap='Electric', title='past_traj. idx {}'.format(idx)))
            # self.vis.heatmap(X=feat[2], opts=dict(colormap='Electric', title='red. idx {}'.format(idx)))
            # self.vis.heatmap(X=feat[3], opts=dict(colormap='Electric', title='green. idx {}'.format(idx)))
            # self.vis.heatmap(X=feat[4], opts=dict(colormap='Electric', title='blue. idx {}'.format(idx)))

    def test_loader(self):
        grid_size = 60
        n_traj = 10
        loader = OffroadLoader(grid_size=grid_size, n_traj=n_traj, train=False)
        loader = DataLoader(loader, num_workers=1, batch_size=1, shuffle=True)
        for idx, (feat, past_traj, future_traj) in enumerate(loader):
            feat = torch.squeeze(feat,dim=0)
            past_traj = torch.squeeze(past_traj,dim=0)
            future_traj = torch.squeeze(future_traj,dim=0)
            self.vis.heatmap(X=feat[0], opts=dict(colormap='Electric', title='height_max. idx {}'.format(idx)))
            self.vis.heatmap(X=feat[1], opts=dict(colormap='Electric', title='height_var. idx {}'.format(idx)))
            overlay_map = overlay_traj_to_map(past_traj, feat[0], 3)
            overlay_map = overlay_traj_to_map(future_traj, overlay_map, 5)
            self.vis.heatmap(X=overlay_map, opts=dict(colormap='Electric', title='overlayed. idx {}'.format(idx)))
            self.vis.heatmap(X=feat[5], opts=dict(colormap='Electric', title='inferred_traj. idx {}'.format(idx)))
            self.vis.heatmap(X=feat[2], opts=dict(colormap='Electric', title='red. idx {}'.format(idx)))
            self.vis.heatmap(X=feat[3], opts=dict(colormap='Electric', title='green. idx {}'.format(idx)))
            self.vis.heatmap(X=feat[4], opts=dict(colormap='Electric', title='blue. idx {}'.format(idx)))


if __name__ == '__main__':
    logging.basicConfig(filename='test_offroad_loader_v2.log', format='%(levelname)s. %(asctime)s. %(message)s',
                        level=logging.DEBUG)
    unittest.main()
