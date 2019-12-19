"""
Copyright SenseTime.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import sys
sys.path.insert(0, '../')
from posenet.PoseNet import PoseNet
from attentionmapnet.SEAttentionPoseNet import SEAttentionPoseNet
from config import ExperimentConfigurator
from common.evaluator import Evaluator
from common.var_evaluator import VarEvaluator
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
    parser.add_argument('--var', action='store_true')

    args = parser.parse_args()

    return args

def parse_args(configurator, args=None):
    if args==None or args.config == None:
        configurator.set_default_opts()
        configurator.process_with_params()
    else:
        configurator.load_from_json(args.config)

    if "uniform_sampling" not in configurator.opt:
	print("no uniform_sampling")
	configurator.opt["uniform_sampling"] = False
    else:
	print("uniform_sampling exists.")
    configurator.opt["eval_times"] = 5
    configurator.opt.shuffle = False
    configurator.opt.write_json = False
    configurator.opt.display = False
    configurator.opt.batch_size = 64
    #configurator.opt.eval_checkpoint = "epoch_0100.pth.tar"
    configurator.opt.log_file = "eval_log.txt"
    configurator.set_environmental_variables()
    configuration = configurator.opt
    configurator.print_opt()

    return configuration
if __name__ == "__main__":
    args = get_args()

    # configuration
    configuration = parse_args(ExperimentConfigurator(), args)

    # model
    feature_extractor = models.resnet34(pretrained=False)
    if configuration.model == "mapnet" or configuration.model == "posenet":
        model = PoseNet(feature_extractor, drop_rate=configuration.dropout)
    else:
        model = SEAttentionPoseNet(resnet=feature_extractor, config=configuration, drop_rate=configuration.dropout)

    # data
    dataloader = get_dataloader(configuration)

    # read mean and stdev for un-normalizing predictions
    pose_stats_file = osp.join(configuration.preprocessed_data_path, 'pose_stats.txt')
    pose_m, pose_s = np.loadtxt(pose_stats_file)  # mean and stdev

    configuration.dataset_length = len(dataloader.dataset)
    configuration.pose_m = pose_m
    configuration.pose_s = pose_s

    if not args.var:
        evaluator = Evaluator(
            config=configuration,
            model=model,
            dataloader=dataloader
        )
    else:
        evaluator = VarEvaluator(
            config=configuration,
            model=model,
            dataloader=dataloader
        )

    evaluator.run(False)
