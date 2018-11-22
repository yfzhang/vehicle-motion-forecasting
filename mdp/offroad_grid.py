import numpy as np

np.set_printoptions(threshold=np.inf, precision=2)
import logging
from scipy.misc import imresize
from itertools import product
import math
import random
# import matplotlib.pyplot as plt
from numba import jit

import time
from scipy.spatial.distance import directed_hausdorff

class OffroadGrid(object):
    """
    Offroad grid MDP model:
    - 4 connected
    - deterministic

    ^
    |
    | x-axis
    |
    o-------> y-axis

    row major (x,y)
    (2,0), (2,1), (2,2)
    (1,0), (1,1), (1,2)
    (0,0), (0,1), (0,2)
    """

    def __init__(self, grid_size, discount):
        self.actions = ((0, 1), (1, 0), (0, -1), (-1, 0))  # 0->RIGHT, 1->UP, 2->LEFT, 3->DOWN
        # 0->RIGHT, 1->RIGHT-UP, 2->UP, 3->LEFT-UP, 4->LEFT, 5->LEFT-DOWN, 6->DOWN, 7-RIGHT-DOWN
        # self.actions = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))
        self.n_actions = len(self.actions)
        self.n_states = grid_size ** 2
        self.grid_size = grid_size
        self.discount = discount
        self.rewards = None
        self.feat_mat = None

        self.transit_table = self.create_transit_table()
        self.start_idx = self.xy_to_idx((int(grid_size/2), int(grid_size/2)))

    def __str__(self):
        return "OffroadGrid: grid_size {}, discount {}".format(self.grid_size, self.discount)

    def load_feat(self, feat):
        """
        Load feature matrix
        self.feat_mat is defined as (feature_vector, grid_size, grid_size)
        :param feat:
        :return:
        """
        assert feat.shape[-1] == self.grid_size, "loaded feature dimension does not match grid size"
        self.feat_mat = []
        for idx in range(self.n_states):
            x, y = self.idx_to_xy(idx)
            feat_vector = feat[:, x, y]
            self.feat_mat.append(feat_vector)

        self.feat_mat = np.stack(self.feat_mat, axis=0)

    def find_feat_expect(self, traj):
        """
        Compute feature count based on demonstration trajectory and feature matrix
        :param traj:
        :return: feature counts (numpy array, same dimension as feature vector)
        """
        feat_expect = np.zeros(self.feat_mat.shape[-1])
        for xy in traj:
            idx = self.xy_to_idx((xy[0], xy[1]))
            feat_expect += self.feat_mat[idx, :]

        return feat_expect

    def find_demo_svf(self, traj):
        """
        compute state visitation frequency from demostration trajectory
        :param traj: numpy array, (n_length, 2)
        :return: numpy array, (n_states)
        """
        svf = np.zeros(self.n_states)
        for i, xy in enumerate(traj):
            idx = self.xy_to_idx(((xy[0]), xy[1]))
            svf[idx] += 1

        return svf

    def find_svf(self, traj, policy):
        """
        compute state visitation frequency given current optimal policy
        it uses ground truth traj length as guide for how long it should propogate the policy into the future

        :param traj:
        :param policy: numpy array, (n_states, n_actions)
        :return: numpy array, (n_states)
        """
        traj_len = traj.shape[0]
        start_state = np.zeros(self.n_states)
        start_idx = self.xy_to_idx((traj[0, 0], traj[0, 1]))
        start_state[start_idx] = 1
        svf = np.tile(start_state, (traj_len, 1)).T  # (n_states, n_traj_len)

        for t in range(1, traj_len):
            svf[:, t] = 0
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    svf[self.transit_table[s, a], t] += (svf[s, t - 1] * policy[s, a])

        return svf.sum(axis=1)

    # @jit(parallel=True)
    def find_svf_demo(self, policy, traj_len):
        """
        compute state visitation frequency given current optimal policy
        it uses past_traj length as a guide for how long it should propagate the policy into the future
        it is for the specifically for online inference

        :param policy: numpy array, (n_states, n_actions)
        :param traj_len: int, past traj length
        :return: numpy array, (n_states)
        """
        start_state = np.zeros(self.n_states)
        start_idx = self.xy_to_idx((40, 40))  # always start from the center (40,40)
        start_state[start_idx] = 1
        svf = np.tile(start_state, (traj_len, 1)).T  # (n_states, n_traj_len)

        for t in range(1, traj_len):
            svf[:, t] = 0
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    svf[self.transit_table[s, a], t] += (svf[s, t - 1] * policy[s, a])

        return svf.sum(axis=1)

    def traj_sample(self, policy, traj_len, start_x, start_y):
        """
        """
        def select_action(s, policy, epsilon=0.0):
            prob = np.cumsum(policy[s])
            rand = random.random()
            if rand < prob[0]:
                return 0
            elif rand < prob[1]:
                return 1
            elif rand < prob[2]:
                return 2
            else:
                return 3
        step_count = 0
        current_idx = self.xy_to_idx((start_x, start_y))
        sample_traj = [[start_x, start_y]]
        while step_count < traj_len:
            action = select_action(current_idx, policy)
            current_idx = self.transit_table[current_idx, action]
            sample_traj.append(self.idx_to_xy(current_idx))
            step_count += 1
        return sample_traj


    def find_svf_sample(self, traj, policy, n_samples=1e3, verbose=True):
        """
        compute state visitation frequency given current optimal policy, using sample based method
        :param traj:
        :param policy: numpy array, (n_states, n_actions)
        :return: numpy array, (n_states)
        """
        # import ipdb;ipdb.set_trace()
        start = time.time()
        traj_len = traj.shape[0]
        visitation_count = np.zeros(self.n_states)
        sample_count = 0
        while sample_count < n_samples:
            sample_traj = self.traj_sample(policy, traj_len, traj[0, 0], traj[0, 1])
            for state in sample_traj:
                visitation_count[self.xy_to_idx((state[0],state[1]))] += 1
            sample_count += 1
            # print '  ===> ',self.compute_hausdorff_dist(traj, np.array(sample_traj))
        # print('find_svf_sample. took {} s'.format(time.time() - start))
        return visitation_count / n_samples

    def compute_hausdorff_dist(self, traj1, traj2):
        """
        double side hausdorff distance
        """
        # import ipdb;ipdb.set_trace()
        dist1 = directed_hausdorff(traj1, traj2)[0]
        dist2 = directed_hausdorff(traj2, traj1)[0]
        return max(dist1, dist2)

    def compute_hausdorff_loss(self, policy, demo_traj, n_samples=100):
        """
        :param policy: n_states x 4 numpy array
        :param demo_traj: traj_len x 2 numpy array
        :param n_samples:
        :return: average hausdorff distance
        """
        hau_dist_all = 0.0
        for k in range(n_samples):
            sample_traj = self.traj_sample(policy, demo_traj.shape[0], demo_traj[0,0], demo_traj[0,1])
            hau_dist = self.compute_hausdorff_dist(demo_traj, sample_traj)
            hau_dist_all += hau_dist
        return hau_dist_all/n_samples


    def find_stochastic_policy(self, value, reward):
        """
        find stochastic policy given values and rewards.
        :param value: numpy array, value[n_states]
        :param reward: numpy array, reward[n_states]
        :return: numpy array, probability[n_states, n_actions]
        """
        Q = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_s = self.transit(s, a)
                Q[s, a] = reward[next_s] + self.discount * value[next_s]

        Q -= Q.max(axis=1).reshape((self.n_states, 1))  # For numerical stability
        Q = np.exp(Q*20) / np.exp(Q*20).sum(axis=1).reshape((self.n_states, 1))  # softmax over actions
        return Q

    def find_optimal_value(self, reward, thresh=0.005):
        """
        find optimal value for each state, given rewards. with resetting goal state value to 0.
        :param reward: numpy array (n_states)
        :return:
        """
        start = time.clock()
        value = np.zeros(self.n_states)
        step = 0
        import warnings
        max_update = np.inf
        while max_update > thresh:
            max_update = 0.0
            step += 1

            for s in range(self.n_states):
                next_s_list = [self.transit_table[s, a] for a in range(self.n_actions)]
                new_v = reward[s] + max([self.discount * value[ss] for ss in next_s_list])

                # find the largest update through out the whole sweep over all states
                max_update = max(max_update, abs(value[s] - new_v))
                value[s] = new_v  # async update

            if step > 1000:
                warnings.warn('value iteration does not converge', RuntimeWarning)
                break

        print('find_optimal_value. iter {}, last update {:.2f}, took {:.2f}'.format(step, max_update, time.clock()-start))
        return value

    def select_action(self, s, value, epsilon):
        if random.random()>epsilon:
            return random.randint(0, self.n_actions-1)
        v_list = [0] * self.n_actions
        for ind, a in enumerate(range(self.n_actions)):
            s_next = self.transit_table[s, a]
            v_next = value[s_next]
            v_list[ind] = v_next
        pi = np.exp(np.array(v_list))
        pi = pi / np.sum(pi)
        pi = np.exp(pi*10) / (np.exp(pi*10).sum())
        randsample = random.random()
        pi_cum = np.cumsum(pi)
        # import ipdb; ipdb.set_trace()

        a_ind = np.where(pi_cum>randsample)[0][0]
        # if a_ind is None:
        #     import ipdb; ipdb.set_trace()
        # print pi_cum,pi,v_list, randsample,a_ind
        return a_ind


    def find_optimal_value_mc(self, reward, horizon=50, gamma=0.9, thresh=0.001, epsilon=0.6):
        """
        Using sample based method to solve the MDP
        find optimal value for each state, given rewards. with resetting goal state value to 0.
        :param reward: numpy array (n_states)
        :horizon: sample trajectory length
        :gamma: discount factor
        :thresh: convergence threshold
        :epsilon: epsilon-greedy
        :return:
        """
        value = np.zeros(self.n_states)
        visit_count = np.zeros(self.n_states)
        step = 0
        max_update, max_update_all = 0, 0
        while max_update > thresh or step < 500:
            max_update = 0.0
            step += 1
            trajectory = []
            s = self.start_idx
            for k in range(horizon):
                a = self.select_action(s, value, epsilon)
                s_next = self.transit_table[s, a]
                r = reward[s_next]
                trajectory.append((s,a,s_next,r))
                s = s_next

            r_cummulate = np.array([x[3] for x in trajectory])
            r_cummulate -= r_cummulate.max() # change all the reward to negtive value
            for k in range(horizon-2, -1, -1):
                r_cummulate[k] += r_cummulate[k+1] * gamma

            # # debug
            # print r_cummulate
            # print [act[1] for act in trajectory]

            for (s,a,s_next,r), r_cum in zip(trajectory, r_cummulate):
                visit_count[s] += 1
                value_old = value[s]
                # if visit_count[s]<100:
                value_new = ((visit_count[s]-1)/float(visit_count[s]))*value[s] + 1.0/float(visit_count[s])*r_cum
                # else: # favour new ones
                #     value_new = 0.99*value[s] + 0.01*r_cum
                # print 'updated value:', value_old, value_new, float(visit_count[s]), r_cum
                value[s] = value_new
                max_update = max(abs(value_old-value[s]), max_update)

            max_update_all = max(max_update, max_update_all)


            # import ipdb; ipdb.set_trace()


        print('find_optimal_value. iteration {}, max_update {}'.format(step, max_update_all))

        # # debug
        # plt.subplot(221)
        # plt.imshow(reward.reshape(self.grid_size, self.grid_size))
        # plt.subplot(222)
        # plt.imshow(value.reshape(self.grid_size, self.grid_size))
        # plt.subplot(223)
        # plt.imshow(np.log(visit_count).reshape(self.grid_size, self.grid_size))
        # plt.subplot(224)
        # plt.imshow()
        # plt.show()

        return value

    def find_optimal_value_softmax(self, reward, traj):
        """
        algorithm 9.1 in Ziebart's thesis
        :param reward:
        :param traj:
        :return:
        """
        reward = np.random.rand(self.n_states) * -1
        value = np.nan_to_num(np.ones(self.n_states) * float("-inf"))
        # value = np.empty(self.n_states, dtype=np.float32)
        # value.fill(-np.inf)
        goal_idx = self.xy_to_idx((traj[-1, 0], traj[-1, 1]))
        max_update = np.inf  # max update of value(s) for all states in one sweep
        step = 0
        while max_update > 1e-1:
            value[goal_idx] = 0.0
            max_update = 0.0
            step += 1
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    next_s = self.transit_table[s, a]
                    new_v = value[s].copy()
                    new_v = self.softmax(new_v, reward[next_s] + self.discount * value[next_s])

                max_update = max(max_update, abs(value[s] - new_v))
                value[s] = new_v
            print('{}, {:.8f}'.format(step, max_update))
        print('find_optimal_value_softmax. iteration {}, max_update {}'.format(step, max_update))
        return value

    @staticmethod
    def softmax(x1, x2):
        """
        soft-maximum calculation. algorithm 9.2 in Ziebart's thesis
        :return:
        """
        max_x = max(x1, x2)
        min_x = min(x1, x2)
        return max_x + np.log(1 + np.exp(min_x - max_x))

    def transit(self, s, a):
        """
        output the next state given current state and action. deterministic assumed.
        :param s:
        :param a:
        :return:
        """

        # 0->RIGHT, 1->UP, 2->LEFT, 3->DOWN
        xi, yi = self.idx_to_xy(s)
        xj, yj = self.actions[a]
        next_xy = np.array([xi + xj, yi + yj])
        np.clip(next_xy, [0, 0], [self.grid_size - 1, self.grid_size - 1], out=next_xy)

        next_s = self.xy_to_idx((next_xy[0], next_xy[1]))
        return next_s

    def create_transit_table(self):
        table = np.full((self.n_states, self.n_actions), np.nan, dtype=np.int)
        for s in range(self.n_states):
            for a in range(self.n_actions):
                xi, yi = self.idx_to_xy(s)
                xj, yj = self.actions[a]
                next_xy = np.array([xi + xj, yi + yj])
                np.clip(next_xy, [0, 0], [self.grid_size - 1, self.grid_size - 1], out=next_xy)
                next_s = self.xy_to_idx((next_xy[0], next_xy[1]))
                table[s, a] = next_s
        if np.any(np.isnan(table)):
            raise ValueError('this is NAN in transition table')

        return table

    def get_action(self, xy, xy_next):
        """
        output the action given current state and next state
        :param s:
        :param s_next:
        :return:
        """
        delta_xy = xy_next - xy
        action = None
        for i, item in enumerate(self.actions):
            if np.all(delta_xy == item):
                action = i

        return action

    def compute_nll(self, policy, demo_traj):
        """
        compute NLL of the demo traj given current optimal policy
        :param policy:  numpy array. probability[n_states, n_actions]
        :param demo_traj: numpy array. [x, y]
        :return: NLL value normalized by traj length
        """
        prob = 1.0
        import warnings
        for i in range(demo_traj.shape[0] - 1):
            action = self.get_action(demo_traj[i], demo_traj[i + 1])
            if action is None:
                raise RuntimeError('no action can move from {} to {}'.format(demo_traj[i], demo_traj[i + 1]))

            state = self.xy_to_idx((demo_traj[i, 0], demo_traj[i, 1]))
            prob *= policy[state, action]

        nll = -math.log(prob) / demo_traj.shape[0]
        return nll


    @staticmethod
    def is_neighbour(i, k):
        """
        Output if two grid position are adjacent or not
        Four connected
        :param i: (x,y) tuple
        :param k: (x,y) tuple
        :return: bool
        """
        return abs(i[0] - k[0]) + abs(i[1] - k[1]) <= 1

    def idx_to_xy(self, i):
        """
        Convert state index to x,y coordinates
        ROW MAJOR !!
        Example: for 10*10 grid, idx_to_xy(11)->(1,1)
        :param i:
        :return:
        """
        return i // self.grid_size, i % self.grid_size

    def xy_to_idx(self, p):
        """
        Convert x,y coordinates to state index
        :param p: (x,y) tuple
        :return:
        """
        return p[0] * self.grid_size + p[1]
