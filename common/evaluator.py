"""
Copyright SenseTime.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from common.pose_utils import optimize_poses, quaternion_angular_error, qexp, calc_vos_safe_fc, calc_vos_safe
from common.utils import load_state_dict
from torch.autograd import Variable
from dataset_loader.seven_scenes import SevenScenes
import os
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from tensorboardX import SummaryWriter

from torchvision import models, transforms
import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import numpy as np
import cPickle
import json

class Evaluator(object):
    def __init__(self, config, model, dataloader):

        self.config = config

        if config.cuda:
            model.cuda()
            torch.cuda.manual_seed(config.seed)
        model.eval()
        # self.model = model
        if torch.cuda.device_count() == 1:
            self.model = model
        else:
            print("Parallel data processing...")
            self.model = DataParallel(model)
        torch.manual_seed(config.seed)

        self.dataloader = dataloader

        pose_m, pose_s = np.loadtxt(config.pose_stats_file)  # mean and stdev

        self.config.dataset_length = len(dataloader.dataset)
        self.config.pose_m = pose_m
        self.config.pose_s = pose_s


        if not config.display:
            plt.switch_backend('Agg')

        self.t_criterion = lambda t_pred, t_target: np.linalg.norm(t_pred - t_target)
        self.q_criterion = quaternion_angular_error


        if config.figure_output_dir is not None:
            if not osp.isdir(config.figure_output_dir):
                os.makedirs(config.figure_output_dir)
        self.err_filename = osp.join(osp.expanduser(config.figure_output_dir), '{:s}.json'.format("Errs"))
        self.image_filename = osp.join(osp.expanduser(config.figure_output_dir), '{:s}.png'.format("figure"))
        self.result_filename = osp.join(osp.expanduser(config.figure_output_dir), '{:s}.pkl'.format("result"))
        self.writer = SummaryWriter(log_dir=osp.join(config.figure_output_dir, 'tflog'))

        if self.config.display:
            plt.show(block=True)


    def calc_poses(self):
        pred_poses = np.zeros((self.config.dataset_length, 7))
        targ_poses = np.zeros((self.config.dataset_length, 7))

        for batch_idx, (data, target) in enumerate(self.dataloader):
            if batch_idx % 10 == 0:
                print 'Image {:d} / {:d}'.format(batch_idx * self.config.batch_size, self.config.dataset_length)
            tail_idx = min(
                self.config.dataset_length,
                (batch_idx + 1) * self.config.batch_size
            )
            idx = [idx for idx in xrange(batch_idx * self.config.batch_size, tail_idx)]

            output = self.step_feedfwd(
                data=data,
                model=self.model
            )
            # 1x7
            size = output.size()
            output = output.cpu().data.numpy().reshape((-1, size[-1]))
            target = target.numpy().reshape((-1, size[-1]))

            q = [qexp(p[3:]) for p in output]
            output = np.hstack((output[:, :3], np.asarray(q)))
            q = [qexp(p[3:]) for p in target]
            target = np.hstack((target[:, :3], np.asarray(q)))

            output[:, :3] = (output[:, :3] * self.config.pose_s) + self.config.pose_m
            target[:, :3] = (target[:, :3] * self.config.pose_s) + self.config.pose_m

            pred_poses[idx, :] = output
            targ_poses[idx, :] = target


        return pred_poses, targ_poses

    def load_weight_from(self, ckpt_fn):
        weights_file = osp.expanduser(ckpt_fn)
        if osp.isfile(weights_file):
            loc_func = lambda storage, loc:storage
            checkpoint = torch.load(weights_file, map_location=loc_func)
            load_state_dict(self.model, checkpoint['model_state_dict'])
            print('Loaded weights from {:s}'.format(weights_file))
            print('Loss of model = {}'.format(checkpoint['loss']))
            print("Epoch = {}".format(checkpoint['epoch']))
        else:
            print('Could not load weights from {:s}'.format(weights_file))
            raise EnvironmentError

    def run(self, eval_all=False):
        if eval_all:
            for idx in range(10):
                epoch = (idx + 1) * 10
                ckpt_fn = osp.join(self.config.checkpoint_dir, "epoch_{:03d}.pth.tar".format(epoch))
                self.err_filename = osp.join(osp.expanduser(self.config.figure_output_dir), 'epoch_{:03d}_{:s}.json'.format(epoch, "Errs"))
                self.image_filename = osp.join(osp.expanduser(self.config.figure_output_dir), 'epoch_{:03d}_{:s}.png'.format(epoch, "figure"))
                self.result_filename = osp.join(osp.expanduser(self.config.figure_output_dir), 'epoch_{:03d}_{:s}.pkl'.format(epoch, "result"))
                self.load_weight_from(ckpt_fn)

                pred_poses, targ_poses = self.calc_poses()

                fig = self.visualize(pred_poses, targ_poses)
                self.writer.add_figure("Images/Evaluation_{:03d}".format(epoch), fig)
                self.export(pred_poses, targ_poses, fig)
        else:
            ckpt_fn = osp.join(self.config.checkpoint_dir, self.config.eval_checkpoint)
            self.load_weight_from(ckpt_fn)
            pred_poses, targ_poses = self.calc_poses()

            fig = self.visualize(pred_poses, targ_poses)
            self.writer.add_figure("Images/Evaluation", fig)
            self.export(pred_poses, targ_poses, fig)


    def visualize(self, pred_poses, targ_poses):
        # create figure object
        fig = plt.figure()
        if self.config.dataset != '7Scenes':
            ax = fig.add_subplot(111)
        else:
            ax = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1)

        # plot on the figure object
        # ss = max(1, int(len(dataset) / 1000))  # 100 for stairs
        ss = 1
        # scatter the points and draw connecting line
        x = np.vstack((pred_poses[::ss, 0].T, targ_poses[::ss, 0].T))
        y = np.vstack((pred_poses[::ss, 1].T, targ_poses[::ss, 1].T))
        if self.config.dataset != '7Scenes':  # 2D drawing
            ax.plot(x, y, color='b', linewidth=0.5)
            ax.scatter(x[0, :], y[0, :], color='r', s=0.8)
            ax.scatter(x[1, :], y[1, :], color='g', s=0.8)
        else:
            z = np.vstack((pred_poses[::ss, 2].T, targ_poses[::ss, 2].T))
            for xx, yy, zz in zip(x.T, y.T, z.T):
                ax.plot(xx, yy, zs=zz, c='b', linewidth=0.2)
            ax.scatter(x[0, :], y[0, :], zs=z[0, :], c='r', depthshade=0, s=0.8)
            ax.scatter(x[1, :], y[1, :], zs=z[1, :], c='g', depthshade=0, s=0.8)
            ax.view_init(azim=119, elev=13)

        return fig

    def export(self, pred_poses, targ_poses, fig):
        t_loss = np.asarray([self.t_criterion(p, t) for p, t in zip(pred_poses[:, :3],
                                                                    targ_poses[:, :3])])
        q_loss = np.asarray([self.q_criterion(p, t) for p, t in zip(pred_poses[:, 3:],
                                                                    targ_poses[:, 3:])])

        errs = {
            "Error in translation(median)": "{:3.2f}".format(np.median(t_loss)),
            "Error in translation(mean)": "{:3.2f}".format(np.mean(t_loss)),
            "Error in rotation(median)": "{:3.2f}".format(np.median(q_loss)),
            "Error in rotation(mean)": "{:3.2f}".format(np.mean(q_loss)),
        }
        print(errs)
        with open(self.err_filename, 'w') as out:
            json.dump(errs, out, sort_keys=True, indent=4)
        print '{:s} saved'.format(self.err_filename)

        fig.savefig(self.image_filename)
        print '{:s} saved'.format(self.image_filename)
        with open(self.result_filename, 'wb') as f:
            cPickle.dump({'targ_poses': targ_poses, 'pred_poses': pred_poses}, f)
        print '{:s} saved'.format(self.result_filename)

    def step_feedfwd(self, data, model, **kwargs):
        data_var = Variable(data, requires_grad=False).cuda(async=True)

        output = model(data_var)
        return output

        # if criterion is not None
        #     loss = criterion(output, target_var)
        #     return loss.data[0], output
