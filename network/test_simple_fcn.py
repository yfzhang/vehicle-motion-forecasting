from network.simple_fcn import SimpleFCN
from torch.autograd import Variable
import unittest
import torch


class TestSimpleFCN(unittest.TestCase):
    def test_main(self):
        net = SimpleFCN(input_size=2)
        net.eval()
        input = Variable(torch.randn(1, 2, 20, 20))
        output = net(input)
        print(output.shape)


if __name__ == '__main__':
    unittest.main()
