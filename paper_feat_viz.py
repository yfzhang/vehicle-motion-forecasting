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


def pred(feat, future_traj, net, n_states, model, grid_size):
    feat = feat.float()
    feat_var = Variable(feat)
    r_var, feat_out_var = net(feat_var)

    reward = r_var.data[0,0].numpy()
    feat_out = feat_out_var.data[0].numpy()
    return reward,  feat_out

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
exp = '6.02'
resume = 'step1520-loss0.698543939023789.pth'

model = offroad_grid.OffroadGrid(grid_size, discount)
n_states = model.n_states
n_actions = model.n_actions

loader = offroad_loader.OffroadLoader(grid_size=grid_size, train=True)
loader.data_list.sort()
data_list = loader.data_list
loader = DataLoader(loader, num_workers=1, batch_size=1, shuffle=False)
data_normalization = sio.loadmat('/data/datasets/yanfu/irl_data/train-data-mean-std.mat')

net = HybridDilated(viz=True)
net.init_weights()
checkpoint = torch.load(os.path.join('exp',exp,resume))
net.load_state_dict(checkpoint['net_state'])
net.eval()

for step, (feat, past_traj, future_traj) in tqdm(enumerate(loader)):
    reward, feat_out = pred(feat, future_traj, net, n_states, model, grid_size)

    future_traj_sample = future_traj[0].numpy()  # choose one sample from the batch
    future_traj_sample = future_traj_sample[~np.isnan(future_traj_sample).any(axis=1)]  # remove appended NAN rows
    future_traj_sample = future_traj_sample.astype(np.int64)
    past_traj_sample = past_traj[0].numpy()  # choose one sample from the batch
    past_traj_sample = past_traj_sample[~np.isnan(past_traj_sample).any(axis=1)]  # remove appended NAN rows
    past_traj_sample = past_traj_sample.astype(np.int64)

    base_name = data_list[step].split('/')[-1].split('.')[0]
    if not os.path.exists(os.path.join('/home/yf/git_repo/inverse_reinforcement_learning/paper_featout_viz', base_name)):
        os.makedirs(os.path.join('/home/yf/git_repo/inverse_reinforcement_learning/paper_featout_viz', base_name))
    plt.imsave('paper_featout_viz/{}/reward.png'.format(base_name), reward)
    img = imageio.imread('paper_featout_viz/{}/reward.png'.format(base_name))
    overlay_img = overlay(img, future_traj_sample, past_traj_sample)
    imageio.imwrite('paper_featout_viz/{}/reward.png'.format(base_name), overlay_img)

    plt.imsave('paper_featout_viz/{}/heightmax.png'.format(base_name), feat[0,0])
    img = imageio.imread('paper_featout_viz/{}/heightmax.png'.format(base_name))
    overlay_img = overlay(img, future_traj_sample, past_traj_sample)
    imageio.imwrite('paper_featout_viz/{}/heightmax.png'.format(base_name), overlay_img)

    plt.imsave('paper_featout_viz/{}/heightvar.png'.format(base_name), feat[0,1])
    img = imageio.imread('paper_featout_viz/{}/heightvar.png'.format(base_name))
    overlay_img = overlay(img, future_traj_sample, past_traj_sample)
    imageio.imwrite('paper_featout_viz/{}/heightvar.png'.format(base_name), overlay_img)

    red = (feat[0, 2].numpy() * data_normalization['red_std'] + data_normalization['red_mean']).astype(np.uint8)
    green = (feat[0, 3].numpy() * data_normalization['green_std'] + data_normalization['green_mean']).astype(np.uint8)
    blue = (feat[0, 4].numpy() * data_normalization['blue_std'] + data_normalization['blue_mean']).astype(np.uint8)
    color = np.stack([red, green, blue], axis=2)
    overlay_color = overlay(color, future_traj_sample, past_traj_sample)
    imageio.imwrite('paper_featout_viz/{}/rgb.png'.format(base_name), overlay_color)

    n_layer = feat_out.shape[0]
    for i in range(n_layer):
        plt.imsave('paper_featout_viz/{}/featout-{}.png'.format(base_name, i), feat_out[i])
        img = imageio.imread('paper_featout_viz/{}/featout-{}.png'.format(base_name, i))
        overlay_img = overlay(img, future_traj_sample, past_traj_sample)
        imageio.imwrite('paper_featout_viz/{}/featout-{}.png'.format(base_name, i), overlay_img)