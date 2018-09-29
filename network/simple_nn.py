import torch.nn as nn
import torch.nn.functional as F


class SimpleNN(nn.Module):
    """
    Simple neural network aims to add non-linearity to feature->reward mapping
    No spatial learning involved
    """

    def __init__(self, input_size, hidden_size=10):
        super(SimpleNN, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # the final output reward is a scalar
        )

    def forward(self, x):
        return self.block(x)
