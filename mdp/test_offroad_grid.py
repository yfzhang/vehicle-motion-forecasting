import unittest
import visdom
from irl.mdp.offroad_grid import OffroadGrid
import numpy as np
import numpy.random as rn
from loader.offroad_loader import OffroadLoader, get_max_min_feature, get_mean_feature


def make_random_model():
    grid_size = rn.randint(2, 15)
    discount = rn.uniform(0.0, 1.0)
    return OffroadGrid(grid_size, discount)


class TestOffroadGrid(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vis = visdom.Visdom()

    def test_main(self):
        grid_size = 30
        discount = 0.9
        model = OffroadGrid(grid_size, discount)
        loader = OffroadLoader(grid_size=grid_size)
        feat, traj = loader[1]
        print(traj)
        traj[np.isnan(traj)] = 0.0  # replace nan as 0.0
        self.vis.heatmap(X=traj, opts=dict(colormap='Electric', title='height_mean'))
        feat[np.isnan(feat)] = -10.0  # replace nan as -10.0 (put a proper small number)
        model.load_feat(feat)
        self.vis.heatmap(X=model.feat_mat, opts=dict(colormap='Electric', title='height_mean'))


    def test_prob_sum_to_one(self):
        """
        Tests that sum of transition probabilities is 1
        :return:
        """
        for _ in range(5):
            model = make_random_model()
            self.assertTrue(np.isclose(model.transition_prob.sum(axis=2), 1).all())


if __name__ == '__main__':
    unittest.main()
