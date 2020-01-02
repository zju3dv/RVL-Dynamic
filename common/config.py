"""
Copyright SenseTime.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import numpy as np
import torch
import os
import sys
import os.path as osp

class Config(object):
    """
        Base configurator
    """
    name = None
    model_name = None
    # root_dir = os.path.abspath("../")
    # sys.path.append(root_dir)

    """
        Training
    """
    n_epochs = 301
    batch_size_per_device = 20
    do_val = True
    seed = 7
    num_workers = 5
    snapshot = 10
    val_freq = 10
    shuffle = True
    plot_validation_abs_error = False
    cuda = True
    color_jitter = 0.0

    """
        Optimization
    """
    optimizer = "adam"
    lr = 1e-4
    weight_decay = 0.0005
    max_grad_norm = 5.0

    """
        logging
    """
    visdom = True
    initialize_visdom = True
    print_freq = 20

    """
        hyperparameters
    """
    beta = -3.0
    gamma = -3.0
    dropout = 0.5
    image_size = (256, 341)
    criterion = "CodeCriterion"

    """
        args
    """
    dataset = "7Scenes"
    scene = "chess"
    dataset_path = "/home/drinkingcoder/Dataset/7Scenes"
    learn_beta = True
    experiment_suffix = "exp"
    devices = "0"
    resume_optim = False
    checkpoint = None
    select_best_model = True
    path_mask = False
    best_model_name = "best_model.pth.tar"
    log_output_dir = "logs"
    salience_output_dir = "salience"
    figure_output_dir = "figures"
    attention_output_dir = "attention"
    loss_result_dir = "figures"

    def __init__(self):
        self.experiment_name = '{:s}_{:s}_{:s}_{:s}_{:s}'.format(self.dataset, self.scene, self.model_name, self.name, self.experiment_suffix)
        self.preprocessed_data_path = osp.join("..", "data", self.dataset)

        self.log_output_dir = osp.join(self.log_output_dir, self.experiment_name)
        self.figure_output_dir = osp.join(self.figure_output_dir, self.experiment_name)
        self.loss_result_dir = osp.join(self.loss_result_dir, self.experiment_name)
        self.salience_output_dir = osp.join(self.salience_output_dir, self.experiment_name)
        self.attention_output_dir = osp.join(self.attention_output_dir, self.experiment_name)

        os.environ['CUDA_VISIBLE_DEVICES'] = self.devices
        self.device_count = torch.cuda.device_count()
        self.batch_size = self.batch_size_per_device * self.device_count

        self.print_config()

    def print_config(self):
        print("Experiment: {}".format(self.experiment_name))
        print("-------------------------------------------------------------------------------------------------")
        print("Training:")
        print("\ttraining epochs = {}".format(self.n_epochs))
        print("\tbatch size = {}".format(self.batch_size))
        print("\tdo validation = {}".format(self.do_val))
        print("\ttime seed= {}".format(self.seed))
        print("\tnum workers= {}".format(self.num_workers))
        print("\tvalidation frequency= {}".format(self.val_freq))
        print("\tcolor jitter = {}".format(self.color_jitter))
        print("\t")

        print("Optimization:")
        print("\tOptimizer = {}".format(self.optimizer))
        print("\tbase learning rate = {}".format(self.lr))
        print("\tweight decay = {}".format(self.weight_decay))
        print("\t")

        print("Logging:")
        print("\tVisdom enabled: {}".format(self.visdom))
        print("\tlog print frequency: {}".format(self.print_freq))
        print("\t")

        print("Hyperparamters:")
        print("\tbeta = {}".format(self.beta))
        print("\tgamma = {}".format(self.gamma))
        print("\tdropout = {}".format(self.dropout))
        print("\tcriterion = {}".format(self.criterion))
        print("\t")

        print("Args:")
        print("\tdataset = {}".format(self.dataset))
        print("\tscene = {}".format(self.scene))
        print("\tdataset path = {}".format(self.dataset_path))
        print("\tlearn_beta = {}".format(self.learn_beta))
        print("\tdevices = {} / {}".format(self.devices, self.device_count))
        print("\tcheckpoint = {}".format(self.checkpoint) )
        print("\tsalience_dir: {}".format(self.salience_output_dir))
        print("\tfigure_output_dir: {}".format(self.figure_output_dir))
        print("\tpreprocessed data path: {}".format(self.preprocessed_data_path))
        print("-------------------------------------------------------------------------------------------------")

