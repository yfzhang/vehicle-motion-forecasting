import torch.nn as nn
import torch


class HybridFCN(nn.Module):
    """
    Hybrid FCN with 3*3 filters
    kinematic related information will be directly feed to higher layers
    """

    def __init__(self, feat_in_size=5, feat_out_size=60, regression_hidden_size=32, output_size=1):
        super(HybridFCN, self).__init__()
        self.feat_in_size = feat_in_size

        # receptive field: 3 -> 5 -> 7 -> 9
        self.feat_block = nn.Sequential(
            nn.ReflectionPad2d(1),  # use mirror like padding. good for segmentation.
            nn.Conv2d(feat_in_size, 64, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, feat_out_size, 3),
            nn.ReLU(inplace=True),
        )
        self.regression_block = nn.Sequential(
            nn.Conv2d(feat_out_size + 5, regression_hidden_size, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(regression_hidden_size, regression_hidden_size, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(regression_hidden_size, regression_hidden_size, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(regression_hidden_size, output_size, 1),
        )

    def forward(self, x):
        # geometric and semantic feature extraction
        feat_in = x[:, :self.feat_in_size, :, :]
        feat_out = self.feat_block(feat_in)

        kinematic_in = torch.cat((feat_out, x[:, 5:, :, :]), dim=1)

        out = self.regression_block(kinematic_in)
        return out

    def init_weights(self):
        for name, mod in self.feat_block.named_children():
            if mod.__class__.__name__ == 'Conv2d':
                nn.init.kaiming_normal(mod.weight, a=0)