# supervised learning baseline
# use same network: hybrid_fcn
# Huber loss

import loader.offroad_loader as offroad_loader
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn 
import numpy as np

np.set_printoptions(threshold=np.inf, precision=4)  # print the full numpy array
import visdom
import warnings
import logging
import os
import math

from network.hybrid_fcn import HybridFCN
import torch
import time
# from maxent_nonlinear_offroad import pred, rl, overlay_traj_to_map, visualize

n_epoch = 10
resume = None
lr = 1e-3
grid_size = 80
n_worker = 4
batch_size = 16
showiter = 20
snapshot = 100
exp_name = '1_1_sl'

if not os.path.exists(os.path.join('exp', exp_name)):
    os.makedirs(os.path.join('exp', exp_name))

# host = os.environ['HOSTNAME']
vis = visdom.Visdom(env='main')
# vis = visdom.Visdom(env='v{}'.format(exp_name), server='http://localhost', port=8097)

train_loader = offroad_loader.OffroadLoader(grid_size=grid_size, datadir = '/datadrive')
train_loader = DataLoader(train_loader, num_workers=n_worker, batch_size=batch_size, shuffle=True)
test_loader = offroad_loader.OffroadLoader(grid_size=grid_size, train=False, datadir = '/datadrive')
test_loader = DataLoader(test_loader, num_workers=n_worker, batch_size=batch_size, shuffle=True)

net = HybridFCN()
net.cuda()

opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
criterion = nn.SmoothL1Loss()

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

""" train """
running_loss = 0
count = 0
for epoch in range(n_epoch):
    for step, (feat, past_traj, future_traj) in enumerate(train_loader):
        # import ipdb; ipdb.set_trace()
        # start = time.time()
        net.train()
        # print('main. step {}'.format(step))

        target = np.zeros((feat.size()[0], 1, grid_size, grid_size))
        for s in range(feat.size()[0]):
            for w in range(future_traj.size()[1]):
                locx = future_traj[s,w,0]
                locy = future_traj[s,w,1]
                if (not math.isnan(locx)) and (not math.isnan(locy)):
                    target[s,0, int(locx), int(locy)] = 1
        targetVar = Variable(torch.FloatTensor(target), requires_grad=False)

        outVar = net(Variable(feat.float().cuda(), requires_grad=True))

        loss = criterion(outVar, targetVar.cuda())

        opt.zero_grad()
        loss.backward()
        opt.step()

        # cma. cumulative moving average. window size < 20
        # nll_cma = (nll + nll_cma * min(step, 20)) / (min(step, 20) + 1)
        vis.line(X=np.array([count]), Y=np.array([loss.data[0]]), win=train_loss_win, update='append')
        count += 1

        running_loss += loss.data[0]
        if step % showiter == 0: 
            # import ipdb; ipdb.set_trace()
            pt1 = outVar.data[0,0].cpu().numpy()
            pt2 = target[0,0]
            pt3 = feat[0,0].numpy()
            pt4 = feat[0,3].numpy()
            pt1 = (pt1-np.min(pt1))/(np.max(pt1)-np.min(pt1)+1e-8)
            pt2 = (pt2-np.min(pt2))/(np.max(pt2)-np.min(pt2)+1e-8)
            pt3 = (pt3-np.min(pt3))/(np.max(pt3)-np.min(pt3)+1e-8)
            pt4 = (pt4-np.min(pt4))/(np.max(pt4)-np.min(pt4)+1e-8)
            con_img = np.concatenate((np.concatenate((pt3,pt4), axis=1),(np.concatenate((pt1,pt2), axis=1))), axis=0)
            vis.heatmap(X=con_img, 
                opts=dict(colormap='Electric', title='{} step {}, rewards'.format(epoch, step)))
            print '%d-%d loss: %.4f' % (epoch, step, running_loss/showiter)
            running_loss = 0


        if count % snapshot ==0:
            state = {'step': step, 'net_state': net.state_dict(), 'opt_state': opt.state_dict()}
            path = os.path.join('exp', exp_name, 'step{}-loss{}.pth'.format(step, loss.data[0]))
            torch.save(state, path)


