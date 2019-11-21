import sys
# sys.path.insert(0, '../')
import torch
import torch.nn as nn
# import torch.optim as optim
import torch.nn.functional as F
# from common import pose_utils

class Criterion(nn.Module):
    """
        Notice: Why L1 norm?
        w_abs_t: absolute translation weight
        w_abs_q: absolute queternion weight
        h(p. p*) = ||t - t*|| + beta * ||q - q*||
    """
    def __init__(self, loss_fn=nn.L1Loss(), beta=0.0, gamma=-3.0, learn_beta=False):
        super(Criterion, self).__init__()
        self.loss_fn = loss_fn
        self.learn_beta = learn_beta
        self.beta = nn.Parameter(
            torch.Tensor([beta]),
            requires_grad=learn_beta
        )
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=learn_beta
        )

    def forward(self, pred, target):
        loss = self.loss_fn(pred[:, :3], target[:, :3]) * torch.exp(-self.beta) + self.beta + \
               self.loss_fn(pred[:,3:], target[:,3:]) * torch.exp(-self.gamma) + self.gamma
        """
            Notice: we should start from 3 instead of 4
        """
        return loss


class PoseNet(nn.Module):
    def __init__(self, feature_extractor, drop_rate = 0.5, feat_dim=2048):
        super(PoseNet, self).__init__()
        self.drop_rate = drop_rate

        self.feature_extractor = feature_extractor
        self.feature_extractor.drop_rate = drop_rate

        self.feature_extractor = feature_extractor
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)

        fe_out_planes = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)

        self.fc_xyz  = nn.Linear(feat_dim, 3)
        self.fc_wpqr = nn.Linear(feat_dim, 3)

        init_modules = [self.feature_extractor.fc, self.fc_xyz, self.fc_wpqr]

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = F.relu(x)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)

        xyz = self.fc_xyz(x)
        wpqr = self.fc_wpqr(x)
        return torch.cat((xyz, wpqr), 1)
