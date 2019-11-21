import os.path as osp
import os
import sys
sys.path.insert(0, '../')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms, models

from dataset_loader.dataloaders import get_mapnet_train_dataloader, get_posenet_train_dataloader
from posenet.PoseNet import PoseNet
from posenet.SGPoseNet import SGPoseNet
from posenet.PoseNet import Criterion as PoseNetCriterion
from mapnet.MapNet import MapNet, Criterion
from attentionmapnet.SEAttentionPoseNet import SEAttentionPoseNet
from common.trainer import Trainer
from common.utils import AbsoluteCriterion, AbsolutePoseNetCriterion
from config import ExperimentConfigurator
import argparse

def get_args():
    parser = argparse.ArgumentParser(
        description='Training script for mapnet'
    )

    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--mask_sampling', action='store_true')
    parser.add_argument('--uniform_sampling', action='store_true')
    parser.add_argument('--mask_image', action='store_true')
    parser.add_argument('--new_split', action='store_true')
    parser.add_argument('--sampling_threshold', type=float, default=None)
    parser.add_argument('--devices', type=str, default=None)

    args = parser.parse_args()

    return args

def parse_args(configurator, args=None):
    if args==None or args.config == None:
        configurator.set_default_opts()
    else:
        configurator.load_from_json(args.config)

    if args.model != None:
        configurator.opt.model = args.model
    if args.mask_sampling != None:
        configurator.opt.mask_sampling = args.mask_sampling
    if args.uniform_sampling != None:
        configurator.opt.uniform_sampling = args.uniform_sampling
    if args.mask_image != None:
        configurator.opt.mask_image = args.mask_image
    if configurator.opt.dataset == "NewCambridge":
        configurator.opt.new_split = True
        configurator.opt.mu_mask_name = 'new_mu_mask.jpg'
    if args.sampling_threshold != None:
        configurator.opt.sampling_threshold = args.sampling_threshold
    if args.devices != None:
        configurator.opt.devices = args.devices

    configurator.process_with_params()
    configurator.set_environmental_variables()
    configuration = configurator.opt
    configurator.print_opt()

    return configuration

if __name__ == "__main__":
    args = get_args()

    print("Parse configuration...")
    configuration = parse_args(ExperimentConfigurator() ,args)

    # Model
    print("Load model...")
    #feature_extractor = models.resnet34(pretrained=False)
    feature_extractor = models.resnet34(pretrained=True)
    if configuration.model == 'mapnet':
        posenet = PoseNet(feature_extractor, drop_rate=configuration.dropout)
        model = MapNet(mapnet=posenet)
    elif configuration.model == 'attentionmapnet':
        posenet = SEAttentionPoseNet(resnet=feature_extractor, config=configuration, drop_rate=configuration.dropout)
        model = MapNet(mapnet=posenet)
    elif configuration.model == 'posenet':
        model = PoseNet(feature_extractor, drop_rate=configuration.dropout)
    elif configuration.model == 'attentionposenet':
        model = SEAttentionPoseNet(resnet=feature_extractor, config=configuration, drop_rate=configuration.dropout)
    elif configuration.model == 'sgposenet':
        model = SGPoseNet(resnet=feature_extractor, drop_rate=configuration.dropout)



    param_list = [
        {
            'params': model.parameters()
        }
    ]
    if configuration.model == 'mapnet' or configuration.model == 'attentionmapnet':
        kwargs = dict(
            loss_fn=nn.L1Loss(),
            beta=configuration.beta,
            gamma=configuration.gamma,
            rel_beta=configuration.beta,
            rel_gamma=configuration.gamma,
            learn_beta=configuration.learn_beta,
            learn_rel_beta=configuration.learn_beta
        )
        # train_criterion = Criterion(**kwargs)
        train_criterion = Criterion(**kwargs)
        result_criterion = AbsoluteCriterion()

        # Optimizer
        if configuration.learn_beta:
            param_list.append(
                {
                    'params': [
                        train_criterion.beta,
                        train_criterion.gamma,
                        train_criterion.rel_beta,
                        train_criterion.rel_gamma
                    ]
                }
            )
        # Data
        train_dataloader, valid_dataloader = get_mapnet_train_dataloader(configuration)
    else:
        kwargs = dict(
            loss_fn=nn.L1Loss(),
            beta=configuration.beta,
            gamma=configuration.gamma,
            learn_beta=configuration.learn_beta
        )

        train_criterion = PoseNetCriterion(**kwargs)
        result_criterion = AbsolutePoseNetCriterion()

        if configuration.learn_beta:
            param_list.append(
                {
                    'params': [
                        train_criterion.beta,
                        train_criterion.gamma
                    ]
                }
            )
        train_dataloader, valid_dataloader = get_posenet_train_dataloader(configuration)

    if configuration.optimizer == 'adam':
        optimizer = optim.Adam(param_list, lr=configuration.lr, weight_decay=5e-4)


    # Trainer
    print("Setup trainer...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        configuration=configuration,
        train_criterion=train_criterion,
        val_criterion=train_criterion,
        result_criterion=result_criterion,
        train_dataloader=train_dataloader,
        val_dataloader=valid_dataloader
    )
    trainer.run()
