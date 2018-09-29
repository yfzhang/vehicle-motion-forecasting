import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mdp.offroad_grid as offroad_grid
import loader.offroad_loader as offroad_loader
from torch.utils.data import DataLoader
import numpy as np

from network.hybrid_fcn import HybridFCN
from network.hybrid_dilated import HybridDilated

from torch.autograd import Variable
import torch
import os
from tqdm import tqdm
import scipy.io as sio
import imageio
import time

def overlay(img, future_traj, past_traj):
    overlay_img = img.copy()
    for p in future_traj:
        overlay_img[int(p[0]), int(p[1]), 0] = 255  # red
        overlay_img[int(p[0]), int(p[1]), 1] = 255  # green
        overlay_img[int(p[0]), int(p[1]), 2] = 255  # blue
    for p in past_traj:
        overlay_img[int(p[0]), int(p[1]), 0] = 255
        overlay_img[int(p[0]), int(p[1]), 1] = 0
        overlay_img[int(p[0]), int(p[1]), 2] = 0
    return overlay_img


def pred(feat, future_traj, net, n_states, model, grid_size, past_traj):
    # n_sample = feat.shape[0]
    # start = time.clock()
    feat = feat.float()
    feat_var = Variable(feat)
    r_var = net(feat_var)

    r_sample = r_var[0].data.numpy().squeeze().reshape(n_states)
    future_traj_sample = future_traj[0].numpy()  # choose one sample from the batch
    future_traj_sample = future_traj_sample[~np.isnan(future_traj_sample).any(axis=1)]  # remove appended NAN rows
    future_traj_sample = future_traj_sample.astype(np.int64)
    past_traj_sample = past_traj[0].numpy()  # choose one sample from the batch
    past_traj_sample = past_traj_sample[~np.isnan(past_traj_sample).any(axis=1)]  # remove appended NAN rows
    past_traj_sample = past_traj_sample.astype(np.int64)

    values_sample = model.find_optimal_value(r_sample, 0.1)
    policy = model.find_stochastic_policy(values_sample, r_sample)
    svf_sample = model.find_svf_demo(future_traj_sample, policy, past_traj_sample)
    # print('{} s'.format(time.clock()-start))
    svf = svf_sample.reshape(grid_size, grid_size)
    reward = r_var.data[0, 0].numpy()
    return reward, svf


# initialize param
grid_size = 80
discount = 0.9

exp = '6.34'
resume='step940-loss0.720122159457321.pth'
net = HybridDilated(feat_out_size=25, regression_hidden_size=64)

model = offroad_grid.OffroadGrid(grid_size, discount)
n_states = model.n_states
n_actions = model.n_actions

num = '2.1'

loader = offroad_loader.OffroadLoader(grid_size=grid_size, train=False, demo='demo_data_{}'.format(num), tangent=False)
#loader = offroad_loader.OffroadLoader(grid_size=grid_size, train=False, tangent=False)
loader.data_list.sort()
data_list = loader.data_list
loader = DataLoader(loader, num_workers=1, batch_size=1, shuffle=False)
data_normalization = sio.loadmat('/data/datasets/yanfu/irl_data/train-data-mean-std.mat')

net.init_weights()
checkpoint = torch.load(os.path.join('exp', exp, resume))
net.load_state_dict(checkpoint['net_state'])
net.eval()


root = os.path.join('paper_demo_viz_{}'.format(num), exp)
print(root)

if not os.path.exists(root):
    os.makedirs(root)

for step, (feat, past_traj, future_traj) in enumerate(loader):
    start = time.clock()
    feat = feat.float()
    feat_var = Variable(feat)
    r_var = net(feat_var)
    r_sample = r_var[0].data.numpy().squeeze().reshape(n_states)
    print('{} s: reward network'.format(time.clock()-start))

    start2 = time.clock()
    values_sample = model.find_optimal_value(r_sample, 0.5)
    policy = model.find_stochastic_policy(values_sample, r_sample)
    future_traj_sample = future_traj[0].numpy()  # choose one sample from the batch
    future_traj_sample = future_traj_sample[~np.isnan(future_traj_sample).any(axis=1)]  # remove appended NAN rows
    future_traj_sample = future_traj_sample.astype(np.int64)
    past_traj_sample = past_traj[0].numpy()  # choose one sample from the batch
    past_traj_sample = past_traj_sample[~np.isnan(past_traj_sample).any(axis=1)]  # remove appended NAN rows
    past_traj_sample = past_traj_sample.astype(np.int64)
    svf_sample = model.find_svf_demo(policy, past_traj_sample.shape[0])
    svf = svf_sample.reshape(grid_size, grid_size)
    print('{} s: find svf'.format(time.clock()-start2))
