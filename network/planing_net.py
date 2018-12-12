import torch.nn as nn

# sanity test. use a simple network.
# class PlanningNet(nn.Module):
#     """
#     Hybrid Dilated CNN with 3*3 filters
#     kinematic related information will be directly feed to higher layers
#     """
#
#     def __init__(self, feat_in_size=2, feat_out_size=1, regression_hidden_size=32, viz=False):
#         super(PlanningNet, self).__init__()
#         self.feat_in_size = feat_in_size
#         self.viz = viz
#         self.feat_out_size = feat_out_size
#
#         # receptive filed: 3 -> 7 -> 17 -> 19
#         # dilation rate combination [1, 2, 5] is recommended
#         self.feat_block = nn.Sequential(
#             nn.Conv2d(feat_in_size, 8, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(8, 8, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(8, feat_out_size, 1),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x):
#         # geometric and semantic feature extraction
#         feat_in = x[:, :self.feat_in_size, :, :]
#         out = self.feat_block(feat_in)*(-1) - 1
#
#         return out
#
#     def init_weights(self):
#         for name, mod in self.feat_block.named_children():
#             if mod.__class__.__name__ == 'Conv2d':
#                 nn.init.kaiming_normal(mod.weight, a=0)
#
#     def init_with_pre_train(self, checkpoint):
#         pre_train = checkpoint['net_state']
#         pre_train = {k: v for k, v in pre_train.items() if 'feat_block' in k}
#
#         state_dict = self.state_dict()
#         state_dict.update(pre_train)
#         self.load_state_dict(state_dict)

# sanity test. use a simple network.
class PlanningNet(nn.Module):
    def __init__(self, feat_in_size=5, feat_out_size=1, viz=False):
        super(PlanningNet, self).__init__()
        self.feat_in_size = feat_in_size
        self.viz = viz
        self.feat_out_size = feat_out_size

        # receptive filed: 3 -> 7 -> 17 -> 19
        # dilation rate combination [1, 2, 5] is recommended
        self.feat_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(feat_in_size, 8, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(8, feat_out_size, 3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # geometric and semantic feature extraction
        feat_in = x[:, :self.feat_in_size, :, :]
        out = self.feat_block(feat_in)*(-1) - 1

        return out

    def init_weights(self):
        for name, mod in self.feat_block.named_children():
            if mod.__class__.__name__ == 'Conv2d':
                nn.init.kaiming_normal(mod.weight, a=0)

    def init_with_pre_train(self, checkpoint):
        pre_train = checkpoint['net_state']
        pre_train = {k: v for k, v in pre_train.items() if 'feat_block' in k}

        state_dict = self.state_dict()
        state_dict.update(pre_train)
        self.load_state_dict(state_dict)

# class PlanningNet(nn.Module):
#     """
#     Hybrid Dilated CNN with 3*3 filters
#     kinematic related information will be directly feed to higher layers
#     """
#
#     def __init__(self, feat_in_size=5, feat_out_size=1, regression_hidden_size=32, viz=False):
#         super(PlanningNet, self).__init__()
#         self.feat_in_size = feat_in_size
#         self.viz = viz
#         self.feat_out_size = feat_out_size
#
#         # receptive filed: 3 -> 7 -> 17 -> 19
#         # dilation rate combination [1, 2, 5] is recommended
#         self.feat_block = nn.Sequential(
#             nn.ReflectionPad2d(1),  # use mirror like padding. good for segmentation.
#             nn.Conv2d(feat_in_size, 64, 3),
#             nn.ReLU(inplace=True),
#             nn.ReflectionPad2d(2),
#             nn.Conv2d(64, 64, 3, dilation=2),
#             nn.ReLU(inplace=True),
#             nn.ReflectionPad2d(5),
#             nn.Conv2d(64, 64, 3, dilation=5),
#             nn.ReLU(inplace=True),
#             nn.ReflectionPad2d(1),
#             nn.Conv2d(64, feat_out_size, 3),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x):
#         # geometric and semantic feature extraction
#         feat_in = x[:, :self.feat_in_size, :, :]
#         out = self.feat_block(feat_in)*-1
#
#         return out
#
#     def init_weights(self):
#         for name, mod in self.feat_block.named_children():
#             if mod.__class__.__name__ == 'Conv2d':
#                 nn.init.kaiming_normal(mod.weight, a=0)
#
#     def init_with_pre_train(self, checkpoint):
#         pre_train = checkpoint['net_state']
#         pre_train = {k: v for k, v in pre_train.items() if 'feat_block' in k}
#
#         state_dict = self.state_dict()
#         state_dict.update(pre_train)
#         self.load_state_dict(state_dict)
