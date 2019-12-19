"""
Copyright SenseTime.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from common.config import Config
import os.path as osp
import json
from common.Configurator import Configurator

class ExperimentConfigurator(Configurator):
    def set_default_opts(self):
        super(ExperimentConfigurator, self).set_default_opts()
        self.opt["steps"] = 3
        #self.opt["skip"] = 1
        self.opt["skip"] = 10
        self.opt.batch_size = 64
        self.opt.val_freq = 10
	self.opt.n_epochs = 50
	self.opt.experiment = 'attentiond8_'
        self.opt.dataset_path = "/data/robotcar"
        # self.opt.dataset_path = "/mnt/lustre/huangzhaoyang/Dataset/Cambridge"
	#self.opt.scene = 'loop'
	self.opt.scene = 'full'
	#self.opts.cene = 'detour'
	#self.opt.scene = 'KingsCollege'
	#self.opt.scene = 'ShopFacade'
	#self.opt.scene = 'StMarysChurch'

	# self.opt.image_size = (256, 455)
	#self.opt.dataset = 'Cambridge'
	self.opt.dataset = 'RobotCar'

	self.opt.color_jitter = 0.0
	#self.opt.beta = 0.0 # translation
	# -3, -3 for full
	self.opt.beta = -3.0 # translation
	#self.opt.beta = -3.0 # translation
	self.opt.gamma = -3.0 # translation
	#self.opt.rel_beta = 0.0
        self.opt.mask_sampling = False
	self.opt.uniform_sampling = False
	self.opt.mask_image = False
        self.opt["SE64"] = False
        self.opt["SE128"] = False
        self.opt["SE256"] = False
        self.opt["SE512"] = True
        self.opt["spatial_attention"] = True
        self.opt["sa_ks"] = 1 # spatial attention kernel size
        self.opt.sampling_threshold = 0.05
        self.opt.model = "posenet" # senselocnet
        self.opt.critical_params = [
		'beta',
		'gamma',
		'batch_size',
                'model',
                'mask_sampling',
                'sampling_threshold',
		'color_jitter',
		'uniform_sampling',
		'mask_image',
                'dataset',
		'scene'
            ]

    def process_with_params(self):
	super(ExperimentConfigurator, self).process_with_params()
	if self.opt.dataset == "Cambridge" or self.opt.dataset == "NewCambridge":
	    self.opt.dataset_path = "/mnt/lustre/huangzhaoyang/Dataset/Cambridge"
	    self.opt.image_size = (256, 455)
