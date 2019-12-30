"""
Copyright SenseTime.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import sys
import os.path as osp
import numpy as np

import time
import torch
from torchvision.datasets.folder import default_loader
from torch.utils.data.dataloader import default_collate
import torch.nn as nn
from pose_utils import quaternion_angular_error

def get_configuration(configurator, args=None):
    if args==None or args.config == None:
        configurator.set_default_opts()
        configurator.process_with_params()
    else:
        configurator.load_from_json(args.config)
    configurator.set_environmental_variables()
    configuration = configurator.opt
    configurator.print_opt()

    return configuration

class AbsoluteCriterion(nn.Module):
    """
        w_abs_t: absolute translation weight
        w_abs_q: absolute queternion weight
        h(p. p*) = ||t - t*|| + beta * ||q - q*||
    """
    def __init__(self, loss_fn=nn.L1Loss()):
        super(AbsoluteCriterion, self).__init__()
        self.loss_fn = loss_fn

    def forward(self, pred, target, pose_m, pose_s):
        s = pred.size()
        num_poses = s[0] * s[1]
        pred = pred.view(num_poses, -1)
        target = target.view(num_poses, -1)
        pred[:, :3] = (pred[:, :3] * pose_s) + pose_m
        target[:, :3] = (target[:, :3] * pose_s) + pose_m
        t_loss = self.loss_fn(pred[:, :3], target[:, :3])
        q_loss = self.loss_fn(pred[:, 3:], target[:, 3:])
        return t_loss, q_loss

class AbsolutePoseNetCriterion(nn.Module):
    """
        w_abs_t: absolute translation weight
        w_abs_q: absolute queternion weight
        h(p. p*) = ||t - t*|| + beta * ||q - q*||
    """
    def __init__(self, loss_fn=nn.L1Loss()):
        super(AbsolutePoseNetCriterion, self).__init__()
        self.loss_fn = loss_fn

    def forward(self, pred, target, pose_m, pose_s):
        s = pred.size()
        pred[:, :3] = (pred[:, :3] * pose_s) + pose_m
        target[:, :3] = (target[:, :3] * pose_s) + pose_m
        t_loss = self.loss_fn(pred[:, :3], target[:, :3])
        q_loss = self.loss_fn(pred[:, 3:], target[:, 3:])
        return t_loss, q_loss

def safe_collate(batch):
    batch = filter(lambda x: x is not None, batch)
    return default_collate(batch)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def average(self):
        return self.sum / self.count

class Timer(object):
    def __init__(self):
        self.start_time = 0
        self.diff = 0
        self.avg_meter = AverageMeter()

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.avg_meter.update(self.diff)
        if average:
            return self.avg_meter.avg
        else:
            return self.diff

    def avg_time(self):
        return self.avg_meter.avg

    def last_time(self):
        return self.diff

def load_image(filename, loader=default_loader):
    try:
        img = loader(filename)
    except IOError as e:
        print 'Could not load image {:s}, IOError: {:s}'.format(filename, e)
        return None
    except:
        print 'Could not load image {:s}, unexpected error'.format(filename)
        return None

    return img

def load_state_dict(model, state_dict):
    model_names = [n for n, _ in model.named_parameters()]
    state_names = [n for n in state_dict.keys()]

    # print("mode names:")
    # print(model_names)
    # print("state dict names:")
    # print(state_names)
    state_names[0] = state_names[0].replace('mapnet.', '')
    if model_names[0].find(state_names[0]) >= 0:
        model_prefix = model_names[0].replace(state_names[0], '')
        state_prefix = None
    elif state_names[0].find(model_names[0]) >= 0:
        state_prefix = state_names[0].replace(model_names[0], '')
        model_prefix = None
    else:
        print('Load state dict error: \n\tmodel name:\t{:s}\n\tstate dict name:\t{:s}'.format(
            model_names[0], state_names[0]
        ))
        raise KeyError

    # Notice: What's this!?
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if state_prefix is None:
            k = model_prefix + k
        else:
            k = k.replace(state_prefix, '', 1)
	k = k.replace('mapnet.', '')
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict)

