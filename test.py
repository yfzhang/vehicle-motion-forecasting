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
from multiprocessing import Pool
import os
from maxent_nonlinear_offroad import visualize_batch

# initialize param
grid_size = 80
discount = 0.9
batch_size = 16
n_worker = 8
#exp = '6.24'
#resume = 'step700-loss0.6980162681374217.pth'
#net = HybridDilated(feat_out_size=25, regression_hidden_size=64)

exp = '6.26'
resume = 'step920-loss0.6793519993145658.pth'
net = HybridDilated(feat_out_size=25, regression_hidden_size=64)


def rl(future_traj_sample, r_sample, model, grid_size):
    svf_demo_sample = model.find_demo_svf(future_traj_sample)
    values_sample = model.find_optimal_value_cpp(r_sample, 0.01)
    policy = model.find_stochastic_policy(values_sample, r_sample)
    svf_sample = model.find_svf(future_traj_sample, policy)
    svf_diff_sample = svf_demo_sample - svf_sample
    # (1, n_feature, grid_size, grid_size)
    svf_diff_sample = svf_diff_sample.reshape(1, 1, grid_size, grid_size)
    svf_diff_var_sample = Variable(torch.from_numpy(svf_diff_sample).float(), requires_grad=False)
    nll_sample = model.compute_nll(policy, future_traj_sample)
    dist_sample = model.compute_hausdorff_loss(policy, future_traj_sample, n_samples=1000)
    return nll_sample, svf_diff_var_sample, values_sample, dist_sample


def pred(feat, future_traj, net, n_states, model, grid_size):
    n_sample = feat.shape[0]
    feat = feat.float()
    feat_var = Variable(feat)
    r_var = net(feat_var)

    result = []
    pool = Pool(processes=n_sample)
    for i in range(n_sample):
        r_sample = r_var[i].data.numpy().squeeze().reshape(n_states)
        future_traj_sample = future_traj[i].numpy()  # choose one sample from the batch
        future_traj_sample = future_traj_sample[~np.isnan(future_traj_sample).any(axis=1)]  # remove appended NAN rows
        future_traj_sample = future_traj_sample.astype(np.int64)
        result.append(pool.apply_async(rl, args=(future_traj_sample, r_sample, model, grid_size)))
    pool.close()
    pool.join()
    # extract result and stack svf_diff
    nll_list = [result[i].get()[0] for i in range(n_sample)]
    dist_list = [result[i].get()[3] for i in range(n_sample)]
    svf_diff_var_list = [result[i].get()[1] for i in range(n_sample)]
    values_list = [result[i].get()[2] for i in range(n_sample)]
    svf_diff_var = torch.cat(svf_diff_var_list, dim=0)
    return nll_list, r_var, svf_diff_var, values_list, dist_list


vis = visdom.Visdom(env='test-{}'.format(exp), server='http://128.2.176.221', port=4546)
model = offroad_grid.OffroadGrid(grid_size, discount)
n_states = model.n_states
n_actions = model.n_actions

loader = offroad_loader.OffroadLoader(grid_size=grid_size, train=False)
loader = DataLoader(loader, num_workers=n_worker, batch_size=batch_size, shuffle=False)

net.init_weights()
checkpoint = torch.load(os.path.join('exp', exp, resume))
net.load_state_dict(checkpoint['net_state'])
net.eval()

test_nll_list = []
test_dist_list = []
for step, (feat, past_traj, future_traj) in enumerate(loader):
    start = time.time()
    nll_list, r_var, svf_diff_var, values_list, dist_list = pred(feat, future_traj, net, n_states, model, grid_size)
    test_nll_list += nll_list
    test_dist_list += dist_list
    visualize_batch(past_traj, future_traj, feat, r_var, values_list, svf_diff_var, step, vis, grid_size, train=False)
    print('{}'.format(sum(test_dist_list) / len(test_dist_list)))
nll = sum(test_nll_list) / len(test_nll_list)
dist = sum(test_dist_list) / len(test_dist_list)
vis.text('nll {}'.format(nll))
vis.text('distance {}'.format(dist))
