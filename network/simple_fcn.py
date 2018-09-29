import torch.nn as nn
import torch.nn.functional as F


class SimpleFCN(nn.Module):
    """
    Simple FCN with 1*1 filter to add non-linearity to feature->reward mapping
    No spatial learning involved
    """

    def __init__(self, input_size, hidden_size=32):
        super(SimpleFCN, self).__init__()
        # the last layer should be a conv layer with filter size 1
        self.block = nn.Sequential(
            nn.Conv2d(input_size, hidden_size, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, hidden_size, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, 1, 1),
        )

    def forward(self, x):
        return self.block(x)

    def init_weights(self):
        for name, mod in self.block.named_children():
            if mod.__class__.__name__ == 'Conv2d':
                nn.init.kaiming_normal(mod.weight,a=0)
