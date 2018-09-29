import torch.nn as nn
import torch.nn.functional as F


class StandardFCN(nn.Module):
    """
    Standard FCN with 3*3 filters
    """

    def __init__(self, input_size):
        super(StandardFCN, self).__init__()
        # the last layer should be a conv layer with filter size 1
        self.block = nn.Sequential(
            nn.Conv2d(input_size, 32, 3, padding=1),  # no size change
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1)
        )

    def forward(self, x):
        return self.block(x)

    def init_weights(self):
        for name, mod in self.block.named_children():
            if mod.__class__.__name__ == 'Conv2d':
                nn.init.kaiming_normal(mod.weight, a=0)
