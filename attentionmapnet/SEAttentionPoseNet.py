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

class SEBlock(nn.Module):
    def __init__(self, planes):
        super(SEBlock, self).__init__()
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_features=planes, out_features=planes / 16)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=planes / 16, out_features=planes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.globalAvgPool(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        return out

    def init(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

class SEAttentionPoseNet(nn.Module):
    def __init__(self, resnet, config, drop_rate=0.5, feat_dim=2048):
        super(SEAttentionPoseNet, self).__init__()
        self.spatial_attention = config.spatial_attention
        self.SE = [config.SE64, config.SE128, config.SE256, config.SE512]

        self.drop_rate = drop_rate

        self.resnet = resnet
        self.resnet.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(fe_out_planes, feat_dim)

        self.fc_xyz  = nn.Linear(feat_dim, 3)
        self.fc_wpqr = nn.Linear(feat_dim, 3)

        init_modules = [self.resnet.fc, self.fc_xyz, self.fc_wpqr]
        if self.spatial_attention:
            # self.spatial_attention_module = nn.Linear(512, 1)
            self.spatial_attention_module = nn.Conv2d(512, 1, kernel_size=config.sa_ks, padding=config.sa_ks/2)
            self.sigmoid = nn.Sigmoid()
            init_modules.append(self.spatial_attention_module)

        if self.SE[0]:
            self.SE64 = SEBlock(64)
            init_modules.append(self.SE64)
        if self.SE[1]:
            self.SE128 = SEBlock(128)
            init_modules.append(self.SE128)
        if self.SE[2]:
            self.SE256 = SEBlock(256)
            init_modules.append(self.SE256)
        if self.SE[3]:
            self.SE512 = SEBlock(512)
            init_modules.append(self.SE512)

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Module):
                m.init()

    def forward(self, x):
        # 3x256x256

        x = self.resnet.conv1(x)    # 64x128x128
        x = self.resnet.bn1(x)      # 64x128x128
        x = self.resnet.relu(x)
        # x, attention = self.residual_attention(x)

        x = self.resnet.maxpool(x)  # 64x64x64

        x = self.resnet.layer1(x)   # 64x64x64

        if self.SE[0]:
            attention = self.SE64(x)
            x = x * attention

        x = self.resnet.layer2(x)   # 128x32x32

        if self.SE[1]:
            attention = self.SE128(x)
            x = x * attention

        x = self.resnet.layer3(x)   # 256x16x16

        if self.SE[2]:
            attention = self.SE256(x)
            x = x * attention

        x = self.resnet.layer4(x)   # 512x8x8

        if self.SE[3]:
            attention = self.SE512(x)

            if self.spatial_attention:
                spatial_attention = self.sigmoid(self.spatial_attention_module(x))
                x = x *  spatial_attention
            x = x * attention
        elif self.spatial_attention:
            spatial_attention = self.sigmoid(self.spatial_attention_module(x))
            x = x *  spatial_attention

        x = self.resnet.avgpool(x)  # 512x1x1
        x = x.view(x.size(0), -1)
        x = self.resnet.fc(x)       # 2048

        x = F.relu(x)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)

        xyz = self.fc_xyz(x)
        wpqr = self.fc_wpqr(x)
        # return torch.cat((xyz, wpqr), 1), attention
        return torch.cat((xyz, wpqr), 1)
