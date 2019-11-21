import torch
import torch.nn as nn
import sys
# sys.path.insert(0, '../')
from posenet.PoseNet import PoseNet
from common import pose_utils

class PaperCriterion(nn.Module):
    def __init__(self, loss_fn=nn.L1Loss(), alpha=1.0, beta=0.0, gamma=-3.0, learn_beta=False, learn_rel_beta=False):
        """
        :param loss_fn:
        :param rel_beta: relative position loss weight
        :param rel_gamma: relative quaternion loss weight
        :param beta: position loss weight
        :param gamma: quaternion loss weight
        :param learn_beta: learn beta& gamma or not
        """
        super(PaperCriterion, self).__init__()
        self.loss_fn = loss_fn
        self.beta = nn.Parameter(torch.Tensor([beta]), requires_grad=learn_beta)
        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=learn_beta)
        self.alpha = nn.Parameter(torch.Tensor([alpha]), requires_grad=False)

    def forward(self, pred, targ):
        """
        :param pred: NxTx6
        :param targ: NxTx6
        :return: N
        """
        s = pred.size()
        num_poses = s[0] * s[1]
        abs_loss = \
            torch.exp(-self.beta) * self.loss_fn(
                pred.view(num_poses, -1)[:, :3],
                targ.view(num_poses, -1)[:, :3]
            ) + self.beta + \
            torch.exp(-self.gamma) * self.loss_fn(
                pred.view(num_poses, -1)[:, 3:],
                targ.view(num_poses, -1)[:, 3:]
            ) + self.gamma

        pred_vos = pose_utils.calc_vos_simple(pred)
        targ_vos = pose_utils.calc_vos_simple(targ)

        s = pred.size()
        num_poses = s[0] * s[1]
        rel_loss = \
            torch.exp(-self.beta) * self.loss_fn(
                pred_vos.view(num_poses, -1)[:, :3],
                targ_vos.view(num_poses, -1)[:, :3]
            ) + self.beta + \
            torch.exp(-self.gamma) * self.loss_fn(
                pred_vos.view(num_poses, -1)[:, 3:],
                targ_vos.view(num_poses, -1)[:, 3:]
            ) + self.gamma

        loss = abs_loss + self.alpha * rel_loss
        return loss

class Criterion(nn.Module):
    def __init__(self, loss_fn=nn.L1Loss(), beta=0.0, gamma=-3.0, rel_beta=0.0, rel_gamma=-3.0, learn_beta=False, learn_rel_beta=False):
        """
        :param loss_fn:
        :param rel_beta: relative position loss weight
        :param rel_gamma: relative quaternion loss weight
        :param beta: position loss weight
        :param gamma: quaternion loss weight
        :param learn_beta: learn beta& gamma or not
        """
        super(Criterion, self).__init__()
        self.loss_fn = loss_fn
        self.beta = nn.Parameter(torch.Tensor([beta]), requires_grad=learn_beta)
        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=learn_beta)
        self.rel_beta = nn.Parameter(torch.Tensor([rel_beta]), requires_grad=learn_rel_beta)
        self.rel_gamma = nn.Parameter(torch.Tensor([rel_gamma]), requires_grad=learn_rel_beta)

    def forward(self, pred, targ):
        """
        :param pred: NxTx6
        :param targ: NxTx6
        :return: N
        """
        s = pred.size()
        num_poses = s[0] * s[1]
        abs_loss =\
            torch.exp(-self.beta) * self.loss_fn(
            pred.view(num_poses, -1)[:, :3],
            targ.view(num_poses, -1)[:, :3]
        ) + self.beta + \
            torch.exp(-self.gamma) * self.loss_fn(
                pred.view(num_poses, -1)[:, 3:],
                targ.view(num_poses, -1)[:, 3:]
            ) + self.gamma

        pred_vos = pose_utils.calc_vos_simple(pred)
        targ_vos = pose_utils.calc_vos_simple(targ)

        s = pred_vos.size()
        num_poses = s[0] * s[1]
        rel_loss =\
            torch.exp(-self.rel_beta) * self.loss_fn(
                pred_vos.view(num_poses, -1)[:, :3],
                targ_vos.view(num_poses, -1)[:, :3]
            ) + self.rel_beta +\
            torch.exp(-self.rel_gamma) * self.loss_fn(
                pred_vos.view(num_poses, -1)[:, 3:],
                targ_vos.view(num_poses, -1)[:, 3:]
            ) + self.rel_gamma

        loss = abs_loss + rel_loss
        return loss


class OnlineCriterion(nn.Module):
    def __init__(self, loss_fn=nn.L1Loss(), beta=0.0, gamma=0.0, rel_beta=0.0, rel_gamma=0.0, learn_beta=False, learn_rel_beta=False, gps_mode=False):
        """
        :param loss_fn:
        :param rel_beta: relative position loss weight
        :param rel_gamma: relative quaternion loss weight
        :param beta: position loss weight
        :param gamma: quaternion loss weight
        :param learn_beta: learn beta& gamma or not
        """
        super(OnlineCriterion, self).__init__()
        self.loss_fn = loss_fn
        self.beta = nn.Parameter(torch.Tensor([beta]), requires_grad=learn_beta)
        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=learn_beta)
        self.rel_beta = nn.Parameter(torch.Tensor([rel_beta]), requires_grad=learn_rel_beta)
        self.rel_gamma = nn.Parameter(torch.Tensor([rel_gamma]), requires_grad=learn_rel_beta)
        self.gps_mode = gps_mode

    def forward(self, pred, targ):
        """
        :param pred:  N x 2T x 7,   T absolute poses and T absolute poses(to be computed as relateve poses)
        :param targ:  N x 2T-1 x 7, T absolute poses(ground truth) and T-1 relative poses(from vo)
        :return:
        """
        # absolute poses loss L_D
        s = pred.size()
        T = s[1] / 2
        num_poses = s[0] * T

        pred_abs = pred[:, :T, :].contiguous().view(num_poses, -1)
        targ_abs = targ[:, :T, :].contiguous().view(num_poses, -1)
        abs_loss = torch.exp(-self.beta) * self.loss_fn(
            pred_abs[:, :3],
            targ_abs[:, :3]
        ) + self.beta + \
            torch.exp(-self.gamma) * self.loss_fn(
            pred_abs[:, 3:],
            targ_abs[:, 3:]
        ) + self.gamma

        # vos loss L_T
        pred_vos = pred[:, T:, :].contiguous()
        # should be N x T x 7 when not gps mode
        if not self.gps_mode:
            pred_vos = pose_utils.calc_vos(pred_vos) # compute relative pose from absolute pose
            # should be N x T-1 x 7
            s = pred_vos.size()
            num_poses = s[0] * s[1]
            pred_vos = pred_vos.view(num_poses, -1)
            targ_vos = targ[:, T:, :].contiguous().view(num_poses, -1) # T-1 relative pose

            vos_loss = torch.exp(-self.rel_beta) * self.loss_fn(
                pred_vos[:, :3],
                targ_vos[:, :3]
            ) + self.rel_beta + \
                       torch.exp(-self.rel_gamma) * self.loss_fn(
                pred_vos[:, 3:],
                targ_vos[:, 3:]
            ) + self.rel_gamma
        else: # only 2D position available when gps mode
            s = pred_vos.size()
            num_poses = s[0] * s[1]
            pred_vos = pred_vos.view(num_poses, -1)
            targ_vos = targ[:, T:, :].contiguous().view(num_poses, -1) # T-1 relative pose

            vos_loss = torch.exp(-self.rel_beta) * self.loss_fn(
                pred_vos[:, :2],
                targ_vos[:, :2]
            ) + self.rel_beta

        loss = abs_loss + vos_loss
        return loss

class MapNet(nn.Module):
    def __init__(self, mapnet):
        super(MapNet, self).__init__()
        self.mapnet = mapnet

    def forward(self, x):
        """
        :param x: NxTxCxHxW  --- T(length of window)
        :return: NxTx6
        """
        s = x.size()
        x = x.view(-1, *s[2:]) # arrange layout of input to mapnet
        poses = self.mapnet(x) # maybe we should adjust the batch size
        poses = poses.view(s[0], s[1], -1)
        return poses
