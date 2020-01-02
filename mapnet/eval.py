"""
Copyright SenseTime.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import sys
sys.path.insert(0, '../')
from posenet.PoseNet import PoseNet
from config import MapNetConfigurator
from common.evaluator import Evaluator
from common.utils import get_configuration
import os
import os.path as osp
from dataset_loader.dataloaders import get_test_dataloader as get_dataloader

from torchvision import models, transforms
import torch
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser(
        description='Training script for mapnet'
    )

    parser.add_argument('--config', type=str, default=None)

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_args()

    # configuration
    configuration = get_configuration(MapNetConfigurator(), args)

    # model
    feature_extractor = models.resnet34(pretrained=False)
    model = PoseNet(feature_extractor, drop_rate=configuration.dropout)

    # data
    dataloader = get_dataloader(configuration)

    # read mean and stdev for un-normalizing predictions
    pose_stats_file = osp.join(configuration.preprocessed_data_path, 'pose_stats.txt')
    pose_m, pose_s = np.loadtxt(pose_stats_file)  # mean and stdev

    configuration.dataset_length = len(dataloader.dataset)
    configuration.pose_m = pose_m
    configuration.pose_s = pose_s

    evaluator = Evaluator(
        config=configuration,
        model=model,
        dataloader=dataloader
    )

    evaluator.run()
