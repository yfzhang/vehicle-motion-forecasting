import matplotlib
matplotlib.use('Agg')
import mdp.offroad_grid as offroad_grid
import loader.offroad_loader as offroad_loader
from torch.utils.data import DataLoader
import numpy as np
import visdom

from network.hybrid_fcn import HybridFCN
from network.hybrid_dilated import HybridDilated

from torch.autograd import Variable
import torch
import time
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.io as sio
import imageio
import random


def pred(feat, future_traj, net, n_states, model, grid_size, past_traj):
    # fake a flat open area
    # feat[0, 0] = 0.0
    # feat[0, 1] = -0.65
    feat[0, 0] = random.gauss(-0.5, 0.01)
    feat[0, 1] = random.gauss(-0.5, 0.01)
    feat[0, 2] = 2.4697
    feat[0, 3] = 2.3534
    feat[0, 4] = 2.7742
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

    values_sample = model.find_optimal_value(r_sample, 0.01)
    policy = model.find_stochastic_policy(values_sample, r_sample)
    svf_sample = model.find_svf_demo(future_traj_sample, policy, past_traj_sample)

    svf = svf_sample.reshape(grid_size, grid_size)
    reward = r_var.data[0, 0].numpy()
    return reward, svf


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


# initialize param
grid_size = 80
discount = 0.9

# exp = '6.02'
# resume = 'step740-loss0.7074379453924212.pth'
# net = HybridDilated()

# exp = '6.25'
# resume = 'step980-loss0.7152042616210688.pth'
# net = HybridDilated(feat_out_size=25, regression_hidden_size=64)

# exp = '6.26'
# resume = 'step920-loss0.6793519993145658.pth'
# net = HybridDilated(feat_out_size=25, regression_hidden_size=64)

#exp = '6.33'
#resume='step1840-loss0.7174862543281137.pth'
#net = HybridDilated(feat_out_size=25, regression_hidden_size=64)

exp = '6.34'
resume='step940-loss0.720122159457321.pth'
net = HybridDilated(feat_out_size=25, regression_hidden_size=64)

num = '3'

model = offroad_grid.OffroadGrid(grid_size, discount)
n_states = model.n_states
n_actions = model.n_actions

loader = offroad_loader.OffroadLoader(grid_size=grid_size, train=False, demo='demo_data_{}'.format(num))
#loader = offroad_loader.OffroadLoader(grid_size=grid_size, train=False,)
loader.data_list.sort()
data_list = loader.data_list
loader = DataLoader(loader, num_workers=1, batch_size=1, shuffle=False)
data_normalization = sio.loadmat('/data/datasets/yanfu/irl_data/train-data-mean-std.mat')


net.init_weights()
checkpoint = torch.load(os.path.join('exp', exp, resume))
net.load_state_dict(checkpoint['net_state'])
net.eval()

root = os.path.join('paper_demo_viz_{}'.format(num), 'kinematic-' + exp)

if not os.path.exists(root):
    os.makedirs(root)

for step, (feat, past_traj, future_traj) in tqdm(enumerate(loader)):
    reward, svf = pred(feat, future_traj, net, n_states, model, grid_size, past_traj)
    base_name = data_list[step].split('/')[-1].split('.')[0]

    plt.imsave('{}/{}-reward.png'.format(root, base_name), reward)

    svf_log = np.log(svf + 1e-3)
    plt.imsave('{}/{}-svf.png'.format(root, base_name), svf_log)

    future_traj_sample = future_traj[0].numpy()  # choose one sample from the batch
    future_traj_sample = future_traj_sample[~np.isnan(future_traj_sample).any(axis=1)]  # remove appended NAN rows
    future_traj_sample = future_traj_sample.astype(np.int64)
    past_traj_sample = past_traj[0].numpy()  # choose one sample from the batch
    past_traj_sample = past_traj_sample[~np.isnan(past_traj_sample).any(axis=1)]  # remove appended NAN rows
    past_traj_sample = past_traj_sample.astype(np.int64)

    red = (feat[0, 2].numpy() * data_normalization['red_std'] + data_normalization['red_mean']).astype(np.uint8)
    green = (feat[0, 3].numpy() * data_normalization['green_std'] + data_normalization['green_mean']).astype(np.uint8)
    blue = (feat[0, 4].numpy() * data_normalization['blue_std'] + data_normalization['blue_mean']).astype(np.uint8)
    color = np.stack([red, green, blue], axis=2)
    overlay_color = overlay(color, future_traj_sample, past_traj_sample)
    imageio.imwrite('{}/{}-rgb.png'.format(root, base_name), overlay_color)
