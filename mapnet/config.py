"""
Copyright SenseTime.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from common.config import Config
import os.path as osp
import json
from common.Configurator import Configurator

class MapNetConfigurator(Configurator):
    def set_default_opts(self):
        super(MapNetConfigurator, self).set_default_opts()
        self.opt["steps"] = 3
        self.opt["skip"] = 10
        self.opt.batch_size = 32
        self.opt.val_freq = 10
	self.opt.model = "mapnet"
	self.opt.dataset_path = "/mnt/lustre/huangzhaoyang/Dataset/robotcar"
	self.opt.critical_params = ["batch_size", "mask_sampling", "color_jitter"]
	self.opt.color_jitter = 0.0
	self.opt.mask_sampling = False
	self.opt.sampling_threshold = 0.02
	self.opt.devices = "4, 5, 6, 7"
	# self.opt.devices = "0, 1, 2, 3"

# class MapNetConfig(Config):
#     name = "SenseTime_BatchSize12"
#     model_name = "MapNet"
#     # checkpoint = "logs/7Scenes_chess_MapNet_WeightDecay_and_PaperCriterion/epoch_100.pth.tar"
#     # resume_optim = True
#     dataset = "SenseTime"
#     scene = "Museum"
#     dataset_path = "/home/drinkingcoder/Dataset/sensetime/"
#     # dataset = "7Scenes"
#     # scene = "chess"
#     # dataset_path = "/home/drinkingcoder/Dataset/7Scenes"
#     # num_workers = 1
#
#     batch_size_per_device = 4
#     # snapshot = 1
#     val_freq = 2
#     devices = "0"
#
#     steps = 3
#     skip = 10
#     image_size = (256, 563)
#
#     learn_beta = True
#     criterion = 'CodeCriterion'
#
#     # color_jitter = 0.1
#
#     # experiment_suffix = "continued"
#
#     # def print_config(self):
#
#     def from_json(self, js):
#         self.__dict__.update(js)
#
#     def to_json(self):
#         return json.dumps(self, default=lambda o:o.__dict__)
#
# class MapNetOriginalConfig(Config):
#     name = "CodeCriterion_WeightDecay_BatchSize8_DontLearnBeta"
#     model_name = "MapNet"
#     # checkpoint = "logs/7Scenes_chess_MapNet_WeightDecay_and_PaperCriterion/epoch_100.pth.tar"
#     # resume_optim = True
#     dataset = "RobotCar"
#     scene = "full"
#     dataset_path = "/home/drinkingcoder/Dataset/robotcar"
#     # dataset = "7Scenes"
#     # scene = "chess"
#     # dataset_path = "/home/drinkingcoder/Dataset/7Scenes"
#     # num_workers = 1
#
#     batch_size_per_device = 12
#     # snapshot = 1
#     val_freq = 2
#
#     steps = 3
#     skip = 10
#
#     learn_beta = False
#     criterion = 'CodeCriterion'
#
#     # color_jitter = 0.1
#
#     # experiment_suffix = "continued"
#
#     def print_config(self):
#         super(MapNetConfig, self).print_config()
#
#         print("> Specified Parameters:")
#         print("> \tsteps: {}".format(self.steps))
#         print("> \tskip: {}".format(self.skip))
#
#         print("")
#         print("-------------------------------------------------------------------------------------------------")
#
# class MapNetTestConfig(Config):
#     name = "MapNetTest"
#     model_name = "MapNet"
#     batch_size_per_device = 1
#     devices = "0"
#
#     # checkpoint = "logs/7Scenes_chess_MapNet_CodeCriterion_WeightDecay_BatchSize12_exp/best_model.pth.tar"
#     checkpoint = "logs/7Scenes_chess_MapNet_CodeCriterion_WeightDecay_BatchSize8_DontLearnBeta_exp/best_model.pth.tar"
#     resume_optim = True
#     display = False
#     shuffle = False
#     export_video = True
