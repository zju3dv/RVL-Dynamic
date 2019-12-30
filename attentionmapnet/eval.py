"""
Copyright SenseTime.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import sys
sys.path.insert(0, '../')

from SEAttentionPoseNet import SEAttentionPoseNet
from LearnGAPoseNet import LearnGAPoseNet
# from AvgPoseNet import AvgPoseNet
from config import AttentionMapNetConfigurator
from common.evaluator import Evaluator
import os
import os.path as osp
from dataset_loader.dataloaders import get_test_dataloader as get_dataloader

from common.utils import get_configuration
from torchvision import models, transforms
import torch
import numpy as np
import argparse
from mapnet.MapNet import MapNet

def get_args():
    parser = argparse.ArgumentParser(
        description='Training script for mapnet'
    )

    parser.add_argument('--config', type=str, default=None)

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    # configuration
    configuration = get_configuration(AttentionMapNetConfigurator(), get_args())

    # model
    feature_extractor = models.resnet34(pretrained=True)
    if configuration.model == "SEAttentionPoseNet":
        posenet = SEAttentionPoseNet(resnet=feature_extractor, config=configuration, drop_rate=configuration.dropout)
    elif configuration.model == "LearnGAPoseNet":
        posenet = LearnGAPoseNet(feature_extractor=feature_extractor, drop_rate=configuration.dropout)

    dataloader = get_dataloader(configuration)

    evaluator = Evaluator(
        config=configuration,
        model=posenet,
        dataloader=dataloader
    )

    evaluator.run()
