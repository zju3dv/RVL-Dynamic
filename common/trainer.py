"""
Copyright SenseTime.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import sys
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
from pose_utils import qexp, quaternion_angular_error
from config import Config
from tensorboardX import SummaryWriter

import torch
import torch.utils.data
from torch.nn import DataParallel
from torch.autograd import Variable
import torch.cuda

from utils import Timer, AverageMeter, load_state_dict


class Trainer(object):
    def __init__(self, model, optimizer, configuration, train_criterion, train_dataloader, val_dataloader, val_criterion=None, result_criterion=None, **kwargs):

        self.config = configuration

        if torch.cuda.device_count() == 1:
            self.model = model
        else:
            print("Parallel data processing...")
            self.model = DataParallel(model)
        self.train_criterion = train_criterion

        self.best_model = None
        self.best_model_filename = osp.join(self.config.log_output_dir, self.config.best_model_name)

        if val_criterion is None:
            self.val_criterion = train_criterion
        else:
            self.val_criterion = val_criterion
        if result_criterion is None:
            print("result_criterion is None")
            self.result_criterion = self.val_criterion
        else:
            self.result_criterion = result_criterion

        self.optimizer = optimizer

        if self.config.tf:
            self.writer = SummaryWriter(log_dir=self.config.tf_dir)
            self.loss_win = 'loss_win'
            self.result_win = 'result_win'
            self.criterion_params_win = 'cparam_win'
            criterion_params = {
                k: v.data.cpu().numpy()[0]
                for k, v in self.train_criterion.named_parameters()
            }
            self.n_criterion_params = len(criterion_params)
        # set random seed
        torch.manual_seed(self.config.seed)
        if self.config.cuda:
            torch.cuda.manual_seed(self.config.seed)

        # initiate model with checkpoint
        self.start_epoch = int(1)
        if self.config["checkpoint"]:
            self.load_checkpoint()
        else:
            print("No checkpoint file")
        print('start_epoch = {}'.format(self.start_epoch))

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.pose_m, self.pose_s = np.loadtxt(self.config.pose_stats_file)
        self.pose_m = Variable(torch.from_numpy(self.pose_m).float(), requires_grad=False).cuda(async=True)
        self.pose_s = Variable(torch.from_numpy(self.pose_s).float(), requires_grad=False).cuda(async=True)

        if self.config.cuda:
            self.model.cuda()
            self.train_criterion.cuda()
            self.val_criterion.cuda()

    def run(self):
        n_epochs = self.config.n_epochs
        for epoch in xrange(self.start_epoch, n_epochs + 1):
            # validate
            val_loss = None
            if self.config.do_val and ((epoch % self.config.val_freq == 0) or (epoch == n_epochs - 1)):
                val_loss = self.validate(epoch)
                if self.best_model is None or self.best_model['loss'] is None or val_loss < self.best_model['loss']:
                    self.best_model = self.pack_checkpoint(epoch, val_loss)
                    torch.save(
                        self.best_model,
                        osp.join(self.config.checkpoint_dir, self.config.best_model_name)
                    )
                    print("Best model saved at epoch {:d} in name of {:s}".format(epoch, self.config.best_model_name))

            # save checkpoint
            if epoch % self.config.snapshot == 0:
                checkpoint = self.pack_checkpoint(
                    epoch=epoch,
                    loss=val_loss
                )
                fn = osp.join(self.config.checkpoint_dir, "epoch_{:04d}.pth.tar".format(epoch))
                torch.save(
                    checkpoint,
                    fn
                )
                print('Epoch {:d} checkpoint saved: {:s}'.format(epoch, fn))

            self.train(epoch)

    def get_result_loss(self, output, target):
        target_var = Variable(target, requires_grad=False).cuda(async=True)
        t_loss, q_loss = self.result_criterion(output, target_var, self.pose_m, self.pose_s)

        return t_loss, q_loss

    def train(self, epoch):
        self.model.train()
        train_data_time = Timer()
        train_batch_time = Timer()
        train_data_time.tic()
        for batch_idx, (data, target) in enumerate(self.train_dataloader):
            train_data_time.toc()

            train_batch_time.tic()
            loss, output = self.step_feedfwd(
                data,
                self.model,
                target=target,
                criterion=self.train_criterion,
                optim=self.optimizer,
                train=True
            )

            t_loss, q_loss = self.get_result_loss(output, target)
            train_batch_time.toc()

            if batch_idx % self.config.print_freq == 0:
                n_itr = (epoch - 1) * len(self.train_dataloader) + batch_idx
                epoch_count = float(n_itr) / len(self.train_dataloader)
                print(
                    'Train {:s}: Epoch {:d}\t'
                    'Batch {:d}/{:d}\t'
                    'Data time {:.4f} ({:.4f})\t'
                    'Batch time {:.4f} ({:.4f})\t'
                    'Loss {:f}' .format(
                        self.config.experiment, epoch,
                        batch_idx, len(self.train_dataloader)-1,
                        train_data_time.last_time(), train_data_time.avg_time(),
                        train_batch_time.last_time(), train_batch_time.avg_time(),
                        loss
                    )
                )
                if self.config.tf:
                    self.writer.add_scalars(self.loss_win, {
                        "training_loss":loss
                        }, n_itr)
                    self.writer.add_scalars(self.result_win, {
                        "training_t_loss": t_loss.item(),
                        "training_q_loss": q_loss.item()
                        }, n_itr)
                    if self.n_criterion_params:
                        for name, v in self.train_criterion.named_parameters():
                            v = v.data.cpu().numpy()[0]
                            self.writer.add_scalars(self.criterion_params_win, {
                                    name:v
                                }, n_itr)

            train_data_time.tic()

    def validate(self, epoch):
        # if self.visualize_val_err:
        #     L = len(self.val_dataloader)
        #     # print("L={}".format(L))
        #     batch_size = 10
        #     pred_pose = np.zeros((L * batch_size, 7))
        #     targ_pose = np.zeros((L * batch_size, 7))

        val_batch_time = Timer() # time for step in each batch
        val_loss = AverageMeter()
        t_loss = AverageMeter()
        q_loss = AverageMeter()
        self.model.eval()
        val_data_time = Timer() # time for data retrieving
        val_data_time.tic()
        for batch_idx, (data, target) in enumerate(self.val_dataloader):
            val_data_time.toc()

            val_batch_time.tic()
            loss, output = self.step_feedfwd(
                data,
                self.model,
                target=target,
                criterion=self.val_criterion,
                optim=self.optimizer,           # what will optimizer do in validation?
                train=False
            )
            # NxTx7
            val_batch_time.toc()
            val_loss.update(loss)

            t_loss_batch, q_loss_batch = self.get_result_loss(output, target)
            t_loss.update(t_loss_batch.item())
            q_loss.update(q_loss_batch.item())

            if batch_idx % self.config.print_freq == 0:
                print(
                    'Val {:s}: Epoch {:d}\t'
                    'Batch {:d}/{:d}\t'
                    'Data time {:.4f} ({:.4f})\t'
                    'Batch time {:.4f} ({:.4f})\t'
                    'Loss {:f}'.format(
                        self.config.experiment, epoch,
                        batch_idx, len(self.val_dataloader)-1,
                        val_data_time.last_time(), val_data_time.avg_time(),
                        val_batch_time.last_time(), val_batch_time.avg_time(),
                        loss
                    )
                )

            val_data_time.tic()

        # pred_pose = pred_pose.view(-1, 7)
        # targ_pose = targ_pose.view(-1, 7)
        print('Val {:s}: Epoch {:d}, val_loss {:f}'.format(self.config.experiment, epoch, val_loss.average()))
        print 'Mean error in translation: {:3.2f} m\n' \
              'Mean error in rotation: {:3.2f} degree'.format(t_loss.average(), q_loss.average())

        if self.config.tf:
            n_itr = (epoch - 1) * len(self.train_dataloader)
            self.writer.add_scalars(self.loss_win,{
                "val_loss":val_loss.average()
                }, n_itr)
            self.writer.add_scalars(self.result_win, {
                "val_t_loss": t_loss.average(),
                "val_q_loss": q_loss.average()
                }, n_itr)
            # self.vis.line(
                # X=np.asarray([epoch]),
                # Y=np.asarray([val_loss.average()]),
                # win=self.loss_win,
                # name='val_loss',
                # # append=True,
                # update='append',
                # env=self.vis_env
            # )
            # self.vis.line(
                # X=np.asarray([epoch]),
                # Y=np.asarray([t_loss.average()]),
                # win=self.result_win,
                # name='val_t_loss',
                # update='append',
                # env=self.vis_env
            # )
            # self.vis.line(
                # X=np.asarray([epoch]),
                # Y=np.asarray([q_loss.average()]),
                # win=self.result_win,
                # name='val_q_loss',
                # update='append',
                # env=self.vis_env
            # )
            # self.vis.save(envs=[self.vis_env])

        return t_loss.average()

    def step_feedfwd(self, data, model, target=None, criterion=None, train=True, **kwargs):
        optim = kwargs["optim"]
        if train:
            assert criterion is not None
            data_var = Variable(data, requires_grad=True).cuda(async=True)
            target_var = Variable(target, requires_grad=False).cuda(async=True)
        else:
            data_var = Variable(data, requires_grad=False).cuda(async=True)
            target_var = Variable(target, requires_grad=False).cuda(async=True)

        output = model(data_var)

        if criterion is not None:
            loss = criterion(output, target_var)

            if train:
                optim.zero_grad()
                loss.backward()
                if self.config.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                optim.step()
            return loss.data[0], output
        else:
            return 0, output

        # Help functions
    def load_checkpoint(self):
        checkpoint_file = self.config.checkpoint
        resume_optim = self.config.resume_optim
        if osp.isfile(checkpoint_file):
            loc_func = None if self.config.cuda else lambda storage, loc:storage
            # map_location: specify how to remap storage
            checkpoint = torch.load(checkpoint_file, map_location=loc_func)
            self.best_model = checkpoint
            load_state_dict(self.model, checkpoint["model_state_dict"])

            self.start_epoch = checkpoint['epoch']

            # Is this meaningful !?
            if checkpoint.has_key('criterion_state_dict'):
                c_state = checkpoint['criterion_state_dict']
                # retrieve key in train_criterion
                append_dict = {
                    k: torch.Tensor([0, 0])
                    for k, _ in self.train_criterion.named_parameters()
                    if not k in c_state
                }
                # load zeros into state_dict
                c_state.update(append_dict)
                self.train_criterion.load_state_dict(c_state)

            print("Loaded checkpoint {:s} epoch {:d}".format(
                checkpoint_file, checkpoint['epoch']
            ))
            print("Loss of loaded model = {}".format(checkpoint['loss']))

            if resume_optim:
                print("Load parameters in optimizer")
                self.optimizer.load_state_dict(checkpoint["optim_state_dict"])
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()


            else:
                print("Notice: load checkpoint but didn't load optimizer.")
        else:
            print("Can't find specified checkpoint.!")
            exit(-1)

    def pack_checkpoint(self, epoch, loss=None):
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optimizer.state_dict(),
            'criterion_state_dict': self.train_criterion.state_dict(),
            'loss': loss
        }
        # torch.save(checkpoint_dict, filename)
        return checkpoint_dict

