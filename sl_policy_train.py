# supervised learning baseline
# use same network: hybrid_fcn
# L2 loss

import loader.offroad_loader as offroad_loader
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn 
import torch.nn.functional as f
import numpy as np

np.set_printoptions(threshold=np.inf, precision=4)  # print the full numpy array
import visdom
import os
import math

from network.hybrid_fcn import HybridFCN
import torch
import time
import random
from mdp.offroad_grid import *
# from maxent_nonlinear_offroad import pred, rl, overlay_traj_to_map, visualize

n_epoch = 10
resume = None
lr = 1e-3
grid_size = 80
action_num = 4
n_worker = 4
batch_size = 16
showiter = 50
snapshot = 100
testiter = 50
exp_name = '1_1_sl_policy'
actions = ((0, 1), (1, 0), (0, -1), (-1, 0))

def genTargetVar(future_traj, action_num, grid_size):
    # action direction:
    #     1 ^ x-axis
    #       |
    # 2 <---+---> 0  y-axis
    # action: (0, 1), (1, 0), (0, -1), (-1, 0)
    target = np.zeros((feat.size()[0], action_num, grid_size, grid_size))
    mask = np.zeros((feat.size()[0], action_num, grid_size, grid_size))
    for s in range(feat.size()[0]): # batch
        for w in range(future_traj.size()[1]-1): # trajectory length
            s_x = future_traj[s,w,0]
            s_y = future_traj[s,w,1]
            s_next_x = future_traj[s,w+1,0]
            s_next_y = future_traj[s,w+1,1]
            if math.isnan(s_x) or math.isnan(s_y) or math.isnan(s_next_x) or math.isnan(s_next_y):
                break
            s_x, s_y, s_next_x, s_next_y = int(s_x), int(s_y), int(s_next_x), int(s_next_y)
            mask[s,:, s_x, s_y] = 1
            if s_next_y==s_y+1:
                target[s,0, s_x, s_y] = 1 # action 0
            elif s_next_x==s_x+1:
                target[s,1, s_x, s_y] = 1 # action 1
            elif s_next_y==s_y-1:
                target[s,2, s_x, s_y] = 1 # action 2
            elif s_next_x==s_x-1:
                target[s,3, s_x, s_y] = 1 # action 3
            else:
                print 'trajectory error: (%d,%d)->(%d,%d)' % (s_x, s_y, s_next_x, s_next_y)
    # import ipdb; ipdb.set_trace()
    targetVar = Variable(torch.FloatTensor(target), requires_grad=False)
    maskVar = Variable(torch.FloatTensor(mask), requires_grad=False)
    return targetVar, maskVar


if not os.path.exists(os.path.join('exp', exp_name)):
    os.makedirs(os.path.join('exp', exp_name))

model = OffroadGrid(grid_size, 1.0) # discount is not useful in this 

# host = os.environ['HOSTNAME']
vis = visdom.Visdom(env='main')
# vis = visdom.Visdom(env='v{}'.format(exp_name), server='http://localhost', port=8097)

train_loader = offroad_loader.OffroadLoader(grid_size=grid_size, datadir = '/datadrive')
train_loader = DataLoader(train_loader, num_workers=n_worker, batch_size=batch_size, shuffle=True)
test_loader = offroad_loader.OffroadLoader(grid_size=grid_size, train=False, datadir = '/datadrive')
test_loader = DataLoader(test_loader, num_workers=n_worker, batch_size=batch_size, shuffle=True)

net = HybridFCN(output_size=action_num)
net.cuda()

opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
# criterion = nn.NLLLoss()

if resume is None:
    net.init_weights()
else:
    checkpoint = torch.load(os.path.join('exp', exp_name, resume))
    step = checkpoint['step']
    net.load_state_dict(checkpoint['net_state'])
    opt.load_state_dict(checkpoint['opt_state'])

loss_train, loss_test = 0, 0
train_loss_win = vis.line(X=np.array([-1]), Y=np.array([loss_train]),
                         opts=dict(xlabel='steps', ylabel='loss', title='train acc'))
test_loss_win = vis.line(X=np.array([-1]), Y=np.array([loss_test]),
                        opts=dict(xlabel='steps', ylabel='loss', title='test acc'))
train_nll_win = vis.line(X=np.array([-1]), Y=np.array([0]),
                         opts=dict(xlabel='steps', ylabel='nll', title='train nll'))
test_nll_win = vis.line(X=np.array([-1]), Y=np.array([0]),
                        opts=dict(xlabel='steps', ylabel='nll', title='test nll'))

""" train """
running_loss = 0
count = 0
for epoch in range(n_epoch):
    for step, (feat, past_traj, future_traj) in enumerate(train_loader):
        stepInd = step+1
        # import ipdb; ipdb.set_trace()
        # start = time.time()
        net.train()
        outVar = net(Variable(feat.float().cuda(), requires_grad=True))

        # calculate the loss
        targetVar, maskVar = genTargetVar(future_traj, action_num, grid_size)
        outSoftmaxVar = f.softmax(outVar, dim=1) # used for trajectory sampling
        loss = f.log_softmax(outVar, dim=1) * maskVar.cuda() * targetVar.cuda()
        loss = -loss.sum()

        outForDisplay = outSoftmaxVar.cpu().data.numpy().argmax(axis=1)+1
        targetForDisplay = targetVar.data.numpy().argmax(axis=1)+1
        # loss = criterion(f.log_softmax(outVar * maskVar.cuda(), dim=1), targetVar.squeeze().cuda()) 

        opt.zero_grad()
        loss.backward()
        opt.step()

        # cma. cumulative moving average. window size < 20
        # nll_cma = (nll + nll_cma * min(step, 20)) / (min(step, 20) + 1)
        vis.line(X=np.array([count]), Y=np.array([loss.data[0]]), win=train_loss_win, update='append')
        count += 1

        nll_train = 0
        for k in range(future_traj.size()[0]):
            policy = outSoftmaxVar[k,:,:,:].cpu().data.numpy().reshape(action_num,-1)
            policy = np.transpose(policy,[1,0])
            future_traj_k = future_traj[k].numpy()  # choose one sample from the batch
            future_traj_k = future_traj_k[~np.isnan(future_traj_k).any(axis=1)].astype(np.int32)
            nll = model.compute_nll(policy, future_traj_k)
            nll_train += nll
        vis.line(X=np.array([count]), Y=np.array([nll_train/k]), win=train_nll_win, update='append')

        running_loss += loss.data[0]
        if stepInd % showiter == 0: 
            # import ipdb; ipdb.set_trace()
            # combine several feature map and visualize
            pt1 = outForDisplay[0,:,:]*maskVar[0,0,:,:].data.numpy()
            pt2 = targetForDisplay[0,:,:]
            pt3 = feat[0,0].numpy()
            pt4 = feat[0,3].numpy()
            pt1 = (pt1-np.min(pt1))/4.0
            pt2 = (pt2-np.min(pt2))/4.0
            pt3 = (pt3-np.min(pt3))/(np.max(pt3)-np.min(pt3)+1e-8)
            pt4 = (pt4-np.min(pt4))/(np.max(pt4)-np.min(pt4)+1e-8)
            con_img = np.concatenate((np.concatenate((pt3,pt4), axis=1),(np.concatenate((pt1,pt2), axis=1))), axis=0)
            vis.heatmap(X=con_img, 
                opts=dict(colormap='Electric', title='{} step {}, rewards'.format(epoch, stepInd)))

            # sample trajectories and visualize them
            policy = outSoftmaxVar[0,:,:,:].cpu().data.numpy().reshape(action_num,-1)
            policy = np.transpose(policy,[1,0])
            future_traj_0 = future_traj[0].numpy()  # choose one sample from the batch
            future_traj_0 = future_traj_0[~np.isnan(future_traj_0).any(axis=1)].astype(np.int32)  # remove appended NAN rows
            svf = model.find_svf_sample(future_traj_0,policy, n_samples=1e3, verbose=False)
            vis.heatmap(X=svf.reshape((grid_size,grid_size)), 
                opts=dict(colormap='Electric', title='{} step {}, svf'.format(epoch, stepInd)))

            hou_loss = model.compute_hausdorff_loss(policy, future_traj_0)

            print '%d-%d loss: %.4f, hausdorff loss: %.4f' % (epoch, stepInd, running_loss/showiter, hou_loss)
            running_loss = 0


        if count % snapshot ==0:
            state = {'step': stepInd, 'net_state': net.state_dict(), 'opt_state': opt.state_dict()}
            path = os.path.join('exp', exp_name, 'step{}-loss{}.pth'.format(stepInd, loss.data[0]))
            torch.save(state, path)


        if count % testiter == 0:
            # test
            net.eval()
            loss_test = 0
            nll_test = 0
            for step_test, (feat, past_traj, future_traj) in enumerate(test_loader):
                outVar = net(Variable(feat.float().cuda(), requires_grad=True))

                # calculate the loss
                targetVar, maskVar = genTargetVar(future_traj, action_num, grid_size)
                outSoftmaxVar = f.softmax(outVar, dim=1) # used for trajectory sampling
                loss = f.log_softmax(outVar, dim=1) * maskVar.cuda() * targetVar.cuda()
                loss = -loss.sum()
                loss_test += loss.data[0]

                for k in range(future_traj.size()[0]):
                    policy = outSoftmaxVar[k,:,:,:].cpu().data.numpy().reshape(action_num,-1)
                    policy = np.transpose(policy,[1,0])
                    future_traj_k = future_traj[k].numpy()  # choose one sample from the batch
                    future_traj_k = future_traj_k[~np.isnan(future_traj_k).any(axis=1)].astype(np.int32)
                    nll = model.compute_nll(policy, future_traj_k)
                    nll_test += nll


            vis.line(X=np.array([count]), Y=np.array([loss_test/step_test]), win=test_loss_win, update='append')

            vis.line(X=np.array([count]), Y=np.array([nll_test/step_test/batch_size]), win=test_nll_win, update='append')

