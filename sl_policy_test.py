import loader.offroad_loader as offroad_loader
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as f
import numpy as np

np.set_printoptions(threshold=np.inf, precision=4)  # print the full numpy array
import os

from network.hybrid_fcn import HybridFCN
import torch
from mdp.offroad_grid import *
import matplotlib.pyplot as plt
from tqdm import tqdm

def genTargetVar(future_traj, action_num, grid_size):
    # action direction:
    #     1 ^ x-axis
    #       |
    # 2 <---+---> 0  y-axis
    # action: (0, 1), (1, 0), (0, -1), (-1, 0)
    target = np.zeros((feat.size()[0], action_num, grid_size, grid_size))
    mask = np.zeros((feat.size()[0], action_num, grid_size, grid_size))
    for s in range(feat.size()[0]):  # batch
        for w in range(future_traj.size()[1] - 1):  # trajectory length
            s_x = future_traj[s, w, 0]
            s_y = future_traj[s, w, 1]
            s_next_x = future_traj[s, w + 1, 0]
            s_next_y = future_traj[s, w + 1, 1]
            if math.isnan(s_x) or math.isnan(s_y) or math.isnan(s_next_x) or math.isnan(s_next_y):
                break
            s_x, s_y, s_next_x, s_next_y = int(s_x), int(s_y), int(s_next_x), int(s_next_y)
            mask[s, :, s_x, s_y] = 1
            if s_next_y == s_y + 1:
                target[s, 0, s_x, s_y] = 1  # action 0
            elif s_next_x == s_x + 1:
                target[s, 1, s_x, s_y] = 1  # action 1
            elif s_next_y == s_y - 1:
                target[s, 2, s_x, s_y] = 1  # action 2
            elif s_next_x == s_x - 1:
                target[s, 3, s_x, s_y] = 1  # action 3
            else:
                pass
                # print('trajectory error: ({},{})->({},{})' % (s_x, s_y, s_next_x, s_next_y))
    # import ipdb; ipdb.set_trace()
    targetVar = Variable(torch.FloatTensor(target), requires_grad=False)
    maskVar = Variable(torch.FloatTensor(mask), requires_grad=False)
    return targetVar, maskVar


resume = 'step225-loss311.005523682.pth'
grid_size = 80
action_num = 4
exp_name = '1_1_sl_policy'
actions = ((0, 1), (1, 0), (0, -1), (-1, 0))

if not os.path.exists(os.path.join('exp', exp_name)):
    os.makedirs(os.path.join('exp', exp_name))

model = OffroadGrid(grid_size, 1.0)  # discount is not useful in this

# loader = offroad_loader.OffroadLoader(grid_size=grid_size, demo='demo_data_4', train=False)
loader = offroad_loader.OffroadLoader(grid_size=grid_size, train=False)
loader.data_list.sort()
data_list = loader.data_list
loader = DataLoader(loader, num_workers=1, batch_size=1, shuffle=False)

net = HybridFCN(output_size=action_num)
net.cuda()

if resume is None:
    net.init_weights()
else:
    checkpoint = torch.load(os.path.join('exp', exp_name, resume))
    step = checkpoint['step']
    net.load_state_dict(checkpoint['net_state'])

""" train """
running_loss = 0
count = 0
# test
net.eval()
nll_list = []
dist_list = []
root = os.path.join('paper_demo_viz_4', exp_name)

for step, (feat, past_traj, future_traj) in tqdm(enumerate(loader)):
    outVar = net(Variable(feat.float().cuda(), requires_grad=True))
    # calculate the loss
    targetVar, maskVar = genTargetVar(future_traj, action_num, grid_size)
    outSoftmaxVar = f.softmax(outVar, dim=1)  # used for trajectory sampling
    # loss = f.log_softmax(outVar, dim=1) * maskVar.cuda() * targetVar.cuda()
    # loss = -loss.sum()

    policy = outSoftmaxVar[0, :, :, :].cpu().data.numpy().reshape(action_num, -1)
    policy = np.transpose(policy, [1, 0])
    future_traj_sample = future_traj[0].numpy()  # choose one sample from the batch
    future_traj_sample = future_traj_sample[~np.isnan(future_traj_sample).any(axis=1)].astype(np.int32)
    svf_sample = model.find_svf(future_traj_sample, policy)

    base_name = data_list[step].split('/')[-1].split('.')[0]
    svf = svf_sample.reshape(grid_size, grid_size)
    svf_log = np.log(svf + 1e-3)
    # plt.imsave('{}/{}-svflog-sl.png'.format(root, base_name), svf_log)

    nll = model.compute_nll(policy, future_traj_sample)
    dist = model.compute_hausdorff_loss(policy, future_traj_sample, n_samples=1000)
    nll_list.append(nll)
    print('{}'.format(sum(nll_list) / len(nll_list)))
    dist_list.append(dist)
    print('{}'.format(sum(dist_list)/len(dist_list)))
