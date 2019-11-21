from common.config import Config
import os.path as osp

class PoseNetConfig(Config):
    name = "PoseNet"
    model_name = "PoseNet"
    # checkpoint = "logs/7Scenes_chess_PoseNet_PoseNet Configuration_/epoch_014.pth.tar"
    dataset = "RobotCar"
    scene = "full"
    dataset_path = "/home/drinkingcoder/Dataset/robotcar"

    # def __init__(self):
    #     super(PoseNetConfig, self).__init__()
    #
    #     self.print_config()

    # def print_config(self):
    #     super(PoseNetConfig, self).print_config()

        # print("> Specified Parameters:")
        # print("")
        # print("-------------------------------------------------------------------------------------------------")

class PoseNetTestConfig(Config):
    name = "PoseNet"
    model_name = "PoseNet"
    checkpoint = "logs/7Scenes_chess_PoseNet_PoseNet Configuration_/epoch_014.pth.tar"
    batch_size_per_device = 1
    devices = "0"
    # dataset = "7Scenes"
    # scene = "chess"
    # dataset_path = "/home/drinkingcoder/Dataset/"

    export_video = True
    display = True
    shuffle = False
