import mdp.offroad_grid as offroad_grid
import loader.kinematic_loader as kinematic_loader
from torch.utils.data import DataLoader
import numpy as np

np.set_printoptions(threshold=np.inf)  # print the full numpy array
import visdom
import warnings
import logging
import os

warnings.filterwarnings('ignore')
from network.simple_fcn import SimpleFCN
from torch.autograd import Variable
import torch
import time

logging.basicConfig(filename='maxent_kinematic.log', format='%(levelname)s. %(asctime)s. %(message)s',
                    level=logging.DEBUG)

save_per_steps = 10
# resume = 'step8800-loss1.346111536026001.pth'
resume = None
exp_name = '4.0'
grid_size = 30
discount = 0.9
lr = 5e-3
n_train = 100000  # number of training traj

if not os.path.exists(os.path.join('exp', exp_name)):
    os.makedirs(os.path.join('exp', exp_name))

# host = os.environ['HOSTNAME']
# vis = visdom.Visdom(env='v{}-{}'.format(exp_name, host), server='http://128.2.176.221', port=4546)
vis = visdom.Visdom(env='main')
model = offroad_grid.OffroadGrid(grid_size, discount)  ## takes a long time to init
n_states = model.n_states
n_actions = model.n_actions

train_loader = kinematic_loader.KinematicLoader(grid_size=grid_size, n_traj=n_train)
train_loader = DataLoader(train_loader, num_workers=1, batch_size=1, shuffle=True)

net = SimpleFCN(input_size=4)
step = 0
nll_cma = 0
acc_test = 0

if resume is None:
    net.init_weights()
else:
    checkpoint = torch.load(os.path.join('exp', exp_name, resume))
    step = checkpoint['step']
    net.load_state_dict(checkpoint['net_state'])
    nll_cma = checkpoint['nll_cma']

opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
train_acc_win = vis.line(X=np.array([[-1, -1]]), Y=np.array([[nll_cma, nll_cma]]),
                         opts=dict(xlabel='steps', ylabel='loss', title='train acc'))

for _ in range(1):
    for _, (feat, future_traj) in enumerate(train_loader):
        start = time.time()
        net.train()
        print('main. step {}'.format(step))
        feat = feat[:, :, :, :].float()  # use all layers

        feat_var = Variable(feat)
        r_variable = net(feat_var)

        r = r_variable.data.numpy().squeeze().reshape(n_states)  # (n_state)

        traj = torch.squeeze(future_traj, dim=0).numpy()
        svf_demo = model.find_demo_svf(traj)
        values = model.find_optimal_value(r)
        policy = model.find_stochastic_policy(values, r)
        svf = model.find_svf(traj, policy)
        svf_diff = svf_demo - svf
        svf_diff = svf_diff.reshape(1, 1, grid_size, grid_size)
        # (1, n_feature, grid_size, grid_size)
        svf_diff_var = Variable(torch.from_numpy(svf_diff).float(), requires_grad=False)
        nll = model.compute_nll(policy, traj)

        opt.zero_grad()
        # a hack to enable backprop in pytorch with a vector
        # the normally used loss.backward() only works when loss is a scalar
        torch.autograd.backward([r_variable], [-svf_diff_var])  # TODO: why inversed sign works?
        opt.step()

        print('main. loss {}. took {} s'.format(nll, time.time() - start))
        # cma. cumulative moving average. window size < 20
        nll_cma = (nll + nll_cma * min(step, 20)) / (min(step, 20) + 1)
        vis.line(X=np.array([[step, step]]), Y=np.array([[nll, nll_cma]]), win=train_acc_win, update='append')
        if step % save_per_steps == 0:
            vis.heatmap(X=feat[:, 0, :, :].float().view(grid_size, -1),
                        opts=dict(colormap='Electric', title='step {} 1'.format(step)))
            vis.heatmap(X=feat[:, 1, :, :].float().view(grid_size, -1),
                        opts=dict(colormap='Electric', title='step {} 2'.format(step)))
            vis.heatmap(X=feat[:, 2, :, :].float().view(grid_size, -1),
                        opts=dict(colormap='Electric', title='step {} 3'.format(step)))
            vis.heatmap(X=feat[:, 3, :, :].float().view(grid_size, -1),
                        opts=dict(colormap='Electric', title='step {} 4'.format(step)))

            traj_map = np.zeros((grid_size, grid_size))
            for idx in future_traj.numpy().squeeze():
                logging.debug('main. in traj. idx {}'.format(idx))
                traj_map[idx[0], idx[1]] = 1.0
            vis.heatmap(X=traj_map, opts=dict(colormap='Electric', title='step {} traj'.format(step)))

            vis.heatmap(X=r_variable.data.view(grid_size, -1),
                        opts=dict(colormap='Greys', title='step {}, rewards'.format(step)))
            vis.heatmap(X=values.reshape(grid_size, -1),
                        opts=dict(colormap='Greys', title='step {}, value'.format(step)))
            vis.heatmap(X=svf_diff_var.data.view(grid_size, -1),
                        opts=dict(colormap='Greys', title='step {}, SVF_diff'.format(step)))

            # for name, param in net.named_parameters():
            #    if name.endswith('weight'):
            #        vis.histogram(param.data.view(-1), opts=dict(numbins=20))  # weights
            #        vis.histogram(param.grad.data.view(-1), opts=dict(numbins=20))  # grads

            # save weights
            state = {'nll_cma': nll_cma, 'step': step, 'net_state': net.state_dict(), 'opt_state': opt.state_dict()}

            path = os.path.join('exp', exp_name, 'step{}-loss{}.pth'.format(step, nll_cma))
            torch.save(state, path)

        step += 1
