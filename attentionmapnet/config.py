from common.config import Config
from common.Configurator import Configurator
import os.path as osp


class AttentionMapNetConfigurator(Configurator):
    def __init__(self, fn=None):
        super(AttentionMapNetConfigurator, self).__init__(fn)

    def set_default_opts(self):
        super(AttentionMapNetConfigurator, self).set_default_opts()
        self.opt["model"] = "SEAttentionPoseNet"
        self.opt["steps"] = 3
        self.opt["skip"] = 10
        self.opt.batch_size = 64
        self.opt.val_freq = 10
	#self.opt.devices = "0, 1, 2, 3"
	self.opt.devices = "4, 5, 6, 7"
	self.opt.dataset_path = "/mnt/lustre/huangzhaoyang/Dataset/robotcar"

        self.opt["SE64"] = False
        self.opt["SE128"] = False
        self.opt["SE256"] = False
        self.opt["SE512"] = True
        self.opt["spatial_attention"] = True
        self.opt["sa_ks"] = 1 # spatial attention kernel size
	self.opt["sampling_threshold"] = 0.4
	self.opt.mask_sampling = True

        # self.opt.critical_params.append("model")
        self.opt.critical_params = ["model"]
        if self.opt.model == "SEAttentionPoseNet":
            self.opt.critical_params.extend(
                ["SE512", "spatial_attention", "sa_ks", "mask_sampling", "sampling_threshold"]
            )



# class AvgMapNetConfig(Config):
#     name = ""
#     model_name = "AvgMapNet"
#     # checkpoint = "logs/7Scenes_chess_PoseNet_PoseNet Configuration_/epoch_014.pth.tar"
#     # dataset = "RobotCar"
#     # scene = "full"
#     # dataset_path = "/home/drinkingcoder/Dataset/robotcar"
#     dataset = "7Scenes"
#     scene = "chess"
#     dataset_path = "/home/drinkingcoder/Dataset/7Scenes"
#
#     batch_size_per_device = 4
#
#     steps = 3
#     skip = 10
#
#     image_size = (256, 256)
#
#     def print_config(self):
#         super(AvgMapNetConfig, self).print_config()
#
#         print("> Specified Parameters:")
#         print("> \tsteps: {}".format(self.steps))
#         print("> \tskip: {}".format(self.skip))
#
#         print("")
#         print("-------------------------------------------------------------------------------------------------")
#
# class AttentionMapNetConfig(Config):
#     name = "SEAttentionMapNet"
#     model_name = "MapNet_BatchSize20"
#     # checkpoint = "logs/7Scenes_chess_PoseNet_PoseNet Configuration_/epoch_014.pth.tar"
#     dataset = "RobotCar"
#     scene = "full"
#     dataset_path = "/home/drinkingcoder/Dataset/robotcar"
#
#     batch_size_per_device = 1
#     num_workers = 3
#
#     devices = "0"
#
#     steps = 3
#     skip = 10
#
#     color_jitter = 0.1
#     # image_size = (256, 256)
#
#     def print_config(self):
#         super(AttentionMapNetConfig, self).print_config()
#
#         print("> Specified Parameters:")
#         print("> \tsteps: {}".format(self.steps))
#         print("> \tskip: {}".format(self.skip))
#
#         print("")
#         print("-------------------------------------------------------------------------------------------------")
#
# class AttentionMapNetTestConfig(Config):
#     name = ""
#     model_name = "AvgMapNet"
#     batch_size_per_device = 1
#     devices = "0"
#     # image_size = (256, 256)
#
#     dataset = "RobotCar"
#     scene = "full"
#     dataset_path = "/home/drinkingcoder/Dataset/robotcar"
#
#     checkpoint = "logs/RobotCar_full_MapNet_BatchSize20_AttentionMapNet_exp/best_model.pth.tar"
#     display = True
#     shuffle = False
#     export_video = True
