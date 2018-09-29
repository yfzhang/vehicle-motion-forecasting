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

def pred(feat, future_traj, net, n_states, model, grid_size):
    # n_sample = feat.shape[0]
    feat = feat.float()
    feat_var = Variable(feat)
    r_var = net(feat_var)

    r_sample = r_var[0].data.numpy().squeeze().reshape(n_states)
    future_traj_sample = future_traj[0].numpy()  # choose one sample from the batch
    future_traj_sample = future_traj_sample[~np.isnan(future_traj_sample).any(axis=1)]  # remove appended NAN rows
    future_traj_sample = future_traj_sample.astype(np.int64)

    values_sample = model.find_optimal_value(r_sample, 0.01)
    policy = model.find_stochastic_policy(values_sample, r_sample)
    svf_sample = model.find_svf(future_traj_sample, policy)

    svf = svf_sample.reshape(grid_size, grid_size)
    reward = r_var.data[0,0].numpy()
    return reward,  svf

# initialize param
grid_size = 80
discount = 0.9
exp = '6.02'
resume = 'step740-loss0.7074379453924212.pth'

vis = visdom.Visdom(env='main')
model = offroad_grid.OffroadGrid(grid_size, discount)
n_states = model.n_states
n_actions = model.n_actions

loader = offroad_loader.OffroadLoader(grid_size=grid_size, train=False)
loader.data_list.sort()
data_list = loader.data_list
loader = DataLoader(loader, num_workers=1, batch_size=1, shuffle=False)

net = HybridDilated()
net.init_weights()
checkpoint = torch.load(os.path.join('exp',exp,resume))
net.load_state_dict(checkpoint['net_state'])
net.eval()

for step, (feat, past_traj, future_traj) in tqdm(enumerate(loader)):
    start = time.time()
    reward, svf = pred(feat, future_traj, net, n_states, model, grid_size)
    base_name = data_list[step].split('/')[-1].split('.')[0]
    plt.imsave('paper_dataset_viz/{}-reward.png'.format(base_name), reward)
    svf_log = np.log(svf + 1e-3)
    plt.imsave('paper_dataset_viz/{}-svflog.png'.format(base_name), svf_log)