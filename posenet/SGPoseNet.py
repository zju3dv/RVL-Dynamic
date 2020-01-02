"""
Copyright SenseTime.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import sys
sys.path.insert(0, '../')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from common import pose_utils
from torchvision.models import resnet34
from common.basic_layer import ResidualAttentionBlock

class SGPoseNet(nn.Module):
    def __init__(self, resnet, drop_rate=0.5, feat_dim=2048):
        super(SGPoseNet, self).__init__()

        self.drop_rate = drop_rate
        # self.residual_attention = ResidualAttentionBlock(64, 64)

        self.resnet = resnet
        self.pool8x8 = nn.AdaptiveAvgPool2d(8)
        self.conv = nn.Conv2d(512, 512, 8, 1, bias=False)
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        # self.resnet.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(fe_out_planes, feat_dim)

        self.fc_xyz  = nn.Linear(feat_dim, 3)
        self.fc_wpqr = nn.Linear(feat_dim, 3)

        init_modules = [self.conv, self.bn, self.resnet.fc, self.fc_xyz, self.fc_wpqr]
        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        # 3x256x256

        x = self.resnet.conv1(x)    # 64x128x128
        x = self.resnet.bn1(x)      # 64x128x128
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)  # 64x6SG4x64

        x = self.resnet.layer1(x)   # 64x64x64
        x = self.resnet.layer2(x)   # 128x32x32
        x = self.resnet.layer3(x)   # 256x16x16
        x = self.resnet.layer4(x)   # 512x8x8

        x = self.pool8x8(x)         # 512x3x3
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        # x = self.resnet.avgpool(x)  # 512x1x1
        x = x.view(x.size(0), -1)
        x = self.resnet.fc(x)       # 2048

        x = F.relu(x)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)

        xyz = self.fc_xyz(x)
        wpqr = self.fc_wpqr(x)
        # return torch.cat((xyz, wpqr), 1), attention
        return torch.cat((xyz, wpqr), 1)
