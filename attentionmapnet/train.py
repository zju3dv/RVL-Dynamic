import os.path as osp
import sys
sys.path.insert(0, '../')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms, models

from dataset_loader.composite import MF
from dataset_loader.dataloaders import get_train_transforms as get_transforms
from SEAttentionPoseNet import SEAttentionPoseNet
from LearnGAPoseNet import LearnGAPoseNet
# from AvgPoseNet import AvgPoseNet
from mapnet.MapNet import MapNet, Criterion
from common.trainer import Trainer
from common.utils import AbsoluteCriterion
from config import AttentionMapNetConfigurator
from common.utils import get_configuration
from dataset_loader.dataloaders import get_mapnet_train_dataloader
import argparse
# from config import AvgMapNetConfig

if __name__ == "__main__":
    print("Parse configuration...")
    configuration = get_configuration(AttentionMapNetConfigurator())

    # Model
    print("Load model...")
    feature_extractor = models.resnet34(pretrained=True)
    if configuration.model == "SEAttentionPoseNet":
        posenet = SEAttentionPoseNet(resnet=feature_extractor, config=configuration, drop_rate=configuration.dropout)
    elif configuration.model == "LearnGAPoseNet":
        posenet = LearnGAPoseNet(feature_extractor=feature_extractor, drop_rate=configuration.dropout)

    # posenet = AvgPoseNet(feature_extractor=feature_extractor, drop_rate=configuration.dropout)
    model = MapNet(mapnet=posenet)

    param_list = [
        {
            'params': model.parameters()
        }
    ]
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

    if configuration.optimizer == 'adam':
        optimizer = optim.Adam(param_list, lr=configuration.lr, weight_decay=5e-4)

    # Data
    train_dataloader, valid_dataloader = get_mapnet_train_dataloader(configuration)

    # Trainer
    print("Setup trainer...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        configuration=configuration,
        train_criterion=train_criterion,
        val_criterion=train_criterion,
        result_criterion=AbsoluteCriterion(),
        train_dataloader=train_dataloader,
        val_dataloader=valid_dataloader
    )
    trainer.run()
