"""
Copyright SenseTime.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from PoseNet import PoseNet
from config import PoseNetTestConfig
from common.evaluator import Evaluator
import os
import os.path as osp
from dataset_loader.dataloaders import get_test_dataloader as get_dataloader

from torchvision import models, transforms
import torch
import numpy as np

if __name__ == "__main__":
    # configuration
    configuration = PoseNetTestConfig()

    # model
    feature_extractor = models.resnet34(pretrained=False)
    model = PoseNet(feature_extractor, drop_rate=configuration.dropout)
    # data
    dataloader = get_dataloader(configuration)

    # read mean and stdev for un-normalizing predictions
    pose_stats_file = osp.join(configuration.preprocessed_data_path, configuration.scene, 'pose_stats.txt')
    pose_m, pose_s = np.loadtxt(pose_stats_file)  # mean and stdev

    configuration.dataset_length = len(dataloader)
    configuration.pose_m = pose_m
    configuration.pose_s = pose_s

    evaluator = Evaluator(
        config=configuration,
        model=model,
        dataloader=dataloader
    )

    evaluator.run()
