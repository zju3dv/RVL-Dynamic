from easydict import EasyDict
import numpy as np
import os
import os.path as osp
import sys
import json
from pprint import pprint

class Logger(object):
    def __init__(self, filename='Default.log'):
        self.terminal = sys.stdout
        bufsize = 0
        self.log = open(filename, 'w', bufsize)

    def delink(self):
        self.log.close()

    def writeTerminalOnly(self, message):
        self.terminal.write(message)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class Configurator(object):
    def __init__(self, fn=None):
        if fn:
            self.opt = self.load_from_json(fn)
        else:
            self.opt = EasyDict({})

    def set_default_opts(self):
        self.opt["n_epochs"] = 100
        self.opt["do_val"] = True
        self.opt["seed"] = 7
        self.opt["num_workers"] = 5
        self.opt["snapshot"] = 10
        self.opt["val_freq"] = 10
        self.opt["shuffle"] = True
        self.opt["color_jitter"] = 0.0


        self.opt["optimizer"] = "adam"
        self.opt["lr"] = 1e-4
        self.opt["weight_decay"] = 5e-4
        self.opt["max_grad_norm"] = 5.0

        self.opt["visdom"] = False
        self.opt["tf"] = True
        self.opt["initialize_visdom"] = True
        self.opt["print_freq"] = 20

        self.opt["beta"] = -3.0
        self.opt["gamma"] = -3.0
        self.opt["rel_beta"] = -3.0
        self.opt["rel_gamma"] = -3.0
        self.opt["learn_beta"] = True
        self.opt["dropout"] = 0.5
        self.opt["image_size"] = (256, 341)
        self.opt["cuda"] = True
        self.opt["display"] = True

        self.opt["dataset"] = "RobotCar"
        self.opt["scene"] = "full"
        self.opt["dataset_path"] = "/home/drinkingcoder/Dataset/robotcar"
        self.opt["pose_stats_file"] = "pose_stats.txt"
        self.opt["devices"] = "0"

        self.opt["mask_sampling"] = False
        self.opt["mu_mask_name"] = "mu_mask.jpg"
        self.opt["sampling_threshold"] = 0.05


        self.opt["critical_params"] = [
        ]
        self.opt["experiment"] = "exp_"
        self.opt["model"] = "model"
        self.opt["batch_size"] = 64
        self.opt["figure_output_dir"] = "figures"
        self.opt["checkpoint_dir"] = "ckpts"
        self.opt["best_model_name"] = "best_model.pth.tar"
        self.opt["error_result_dir"] = "error_result"
        self.opt["tf"] = True
        self.opt["tf_dir"] = "tf"
        self.opt["train"] = True
        self.opt["resume_optim"] = False
        self.opt["resume_from_checkpoint"] = False
        self.opt["checkpoint"] = None
        self.opt["export_dir"] = "logs"

        self.opt["write_json"] = True

        self.opt["eval_checkpoint"] = "best_model.pth.tar"
        self.opt["log_file"] = "log.txt"



    def process_with_params(self):
        self.opt["experiment_dir"] = ""

        critical_params = [self.opt[key] for key in self.opt["critical_params"]]
        for name, param in zip(self.opt["critical_params"], critical_params):
            self.opt["experiment_dir"] += "{:s}[{:s}]".format(name, str(param))
        self.opt["experiment_dir"] = self.opt["experiment"] + self.opt["experiment_dir"]

        self.opt["preprocessed_data_path"] = osp.join("..", "data", self.opt.dataset, self.opt.scene)
        self.opt["export_dir"] = osp.join(self.opt["export_dir"], self.opt["experiment_dir"])
        self.opt["log_output_dir"] = self.opt["export_dir"]
        self.opt["figure_output_dir"] = osp.join(self.opt["export_dir"], self.opt["figure_output_dir"])
        self.opt["checkpoint_dir"] = osp.join(self.opt["export_dir"], self.opt["checkpoint_dir"])
        self.opt["tf_dir"] = osp.join(self.opt["export_dir"], self.opt["tf_dir"])
        self.opt["error_result_dir"] = osp.join(self.opt["export_dir"], self.opt["error_result_dir"])
        self.opt["pose_stats_file"] = osp.join(self.opt["preprocessed_data_path"], self.opt["pose_stats_file"])




    def set_environmental_variables(self):
        if not osp.isdir(self.opt["export_dir"]):
            os.makedirs(self.opt["export_dir"])

        if not osp.isdir(self.opt.log_output_dir):
            os.makedirs(self.opt.log_output_dir)
        if not osp.isdir(self.opt.figure_output_dir):
            os.makedirs(self.opt.figure_output_dir)
        if not osp.isdir(self.opt.checkpoint_dir):
            os.makedirs(self.opt.checkpoint_dir)

        os.environ['CUDA_VISIBLE_DEVICES'] = self.opt.devices

        if self.opt["write_json"]:
            fn = osp.join(self.opt.log_output_dir, "config.json")
            self.write_to_json(fn)

        log_file = osp.join(self.opt.log_output_dir, self.opt.log_file)
        print('logging to {:s}'.format(log_file))
        stdout = Logger(log_file)
        sys.stdout = stdout

    def print_opt(self):
        print("Params:")
        print("----------------------------------------------------")
        for key, data in self.opt.items():
            if key not in self.opt["critical_params"]:
                print("\t{:<30s}:{:s}".format(key, str(data)))
        print("----------------------------------------------------")
        print("Critical Params:")
        for key, data in self.opt.items():
            if key in self.opt["critical_params"]:
                print("\t{:<30s}:{:s}".format(key, str(data)))
        print("----------------------------------------------------")

    def load_from_json(self, fn):
        with open(fn, "r") as f:
            jdata = json.load(f)
        self.opt = EasyDict(jdata)
        print("load opt from: {:s}".format(fn))

    def write_to_json(self, fn):
        jdata = json.dumps(self.opt, sort_keys=True, indent=4)
        with open(fn, "w") as f:
            f.write(jdata)
            # json.dump(jdata, f)
        print("write opt to: {:s}".format(fn))
