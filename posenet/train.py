"""
Copyright SenseTime.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os.path as osp
import sys
sys.path.insert(0, '../')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms, models

from dataset_loader.seven_scenes import SevenScenes
from dataset_loader.robotcar import RobotCar
from dataset_loader.sensetime import SenseTime
from dataset_loader.dataloaders import get_train_transforms as get_transforms
from PoseNet import PoseNet, Criterion
from common.utils import AbsoluteCriterion
from common.trainer import Trainer
from config import PoseNetConfig


if __name__ == "__main__":

    print("Parse configuration...")
    configuration = PoseNetConfig()

    # Model
    print("Load model...")
    feature_extractor = models.resnet34(pretrained=True)
    model = PoseNet(feature_extractor, drop_rate=configuration.dropout)
    train_criterion = Criterion(
        beta=configuration.beta,
        learn_beta=configuration.learn_beta
    )
    val_criterion = Criterion(
        learn_beta=False
    ) # beta = 0.0 to see position accuracy

    # Optimizer
    param_list = [
        {
            'params': model.parameters()
        }
    ]
    param_list.append({
        'params': [train_criterion.beta, train_criterion.gamma]
    })

    if configuration.optimizer == 'adam':
        optimizer = optim.Adam(param_list, lr=configuration.lr)

    # Data
    data_transform, target_transform = get_transforms(configuration)

    kwargs = dict(
        scene=configuration.scene,
        data_path=configuration.dataset_path,
        transform=data_transform,
        target_transform=target_transform,
        seed=configuration.seed,
        data_dir=configuration.preprocessed_data_path
    )
    if configuration.dataset == '7Scenes':
        train_data = SevenScenes(train=True,  **kwargs)
        valid_data = SevenScenes(train=False,  **kwargs)
    elif configuration.dataset == 'RobotCar':
        train_data = RobotCar(train=True,  **kwargs)
        valid_data = RobotCar(train=False,  **kwargs)
    elif configuration.dataset == 'SenseTime':
        train_data = SenseTime(train=True, **kwargs)
        valid_data = SenseTime(train=False, **kwargs)
    else:
        raise NotImplementedError

    # Trainer
    print("Setup trainer...")
    pose_stats_file = osp.join(configuration.preprocessed_data_path , configuration.scene, 'pose_stats.txt')

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_criterion=train_criterion,
        val_criterion=val_criterion,
        result_criterion=AbsoluteCriterion(),
        # config_name=args.config_file,
        configuration=configuration,
        experiment=configuration.experiment_name,
        train_dataset=train_data,
        val_dataset=valid_data,
        checkpoint_file=configuration.checkpoint,
        resume_optim=configuration.resume_optim,
        pose_stats_file=pose_stats_file
    )
    trainer.run()
