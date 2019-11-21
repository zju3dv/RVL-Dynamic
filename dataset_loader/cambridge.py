"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import sys
sys.path.insert(0, "../")
import os.path as osp
import torch
from torch.utils import data
import numpy as np
from robotcar_sdk.interpolate_poses import interpolate_vo_poses, \
    interpolate_ins_poses
from robotcar_sdk.camera_model import CameraModel
from robotcar_sdk.image import load_image
import common.utils as utils
from functools import partial
from common.pose_utils import np_qlog_t
import pickle
import cv2
from common.img_utils import read_grayscale_image
from scipy.stats import norm
import transforms3d.quaternions as txq

class Cambridge(data.Dataset):
    def __init__(self, scene, data_path, train, transform=None,
                 target_transform=None, real=False, skip_images=False, seed=7,
                 undistort=False, vo_lib='stereo', data_dir=None, unsupervise=False,
                 config=None):
        """
        :param scene: e.g. 'full' or 'loop'. collection of sequences.
        :param data_path: Root RobotCar data directory.
        Usually '../data/deepslam_data/RobotCar'
        :param train: flag for training / validation
        :param transform: Transform to be applied to images
        :param target_transform: Transform to be applied to poses
        :param real: it determines load ground truth pose or vo pose
        :param skip_images: return None images, only poses
        :param seed: random seed
        :param undistort: whether to undistort images (slow)
        :param vo_lib: Library to use for VO ('stereo' or 'gps')
        (`gps` is a misnomer in this code - it just loads the position information
        from GPS)
        :param data_dir: indicating where to load stats.txt file(to normalize image&pose)
        :param unsupervise: load training set as supervise or unsupervise
        """
        np.random.seed(seed)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.skip_images = skip_images
        self.undistort = undistort
        base_dir = osp.expanduser(osp.join(data_path, scene))
        self.config = config
        # data_dir = osp.join('..', 'data', 'RobotCar', scene)

        if self.config.has_key('new_split') and self.config.new_split:
            print("use new split dataset")
            if train:
                split_filename = osp.join(base_dir, 'train_split.txt')
            else:
                split_filename = osp.join(base_dir, 'test_split.txt')

            with open(split_filename, 'r') as f:
                seqs = [l.rstrip() for l in f if not l.startswith('#')]

            pose_filename = osp.join(base_dir, "dataset_train.txt")
            pose_dict = {}
            with open(pose_filename, 'r') as f:
                data = f.readlines()[3:]

            pose_filename = osp.join(base_dir, "dataset_test.txt")
            with open(pose_filename, 'r') as f:
                data.extend(f.readlines()[3:])

                imgs = [
                    l.split(' ')[0] for l in data if not l.startswith('#')
                ]
                ps = np.asarray([
                    [float(num) for num in l.split(' ')[1:]]
                    for l in data if not l.startswith('#')
                ], dtype=np.float32)

                poses = np.zeros((ps.shape[0], 6))
                poses[:, :3] = ps[:, :3]
                poses[:, 3:] = np_qlog_t(ps[:, 3:])

                for idx, img_name in enumerate(imgs):
                    pose_dict[img_name] = poses[idx, :]

            self.poses = np.empty((0, 6))
            self.imgs = []
            for seq in seqs:
                # seq_dir = osp.join(base_dir, seq)
                img_names = [img for img in imgs if img.startswith(seq)]
                # print(img_names)

                poses = np.asarray([
                    pose_dict[img_name] for img_name in img_names if pose_dict.has_key(img_name)
                ])
                self.imgs.extend([osp.join(base_dir, img_name) for img_name in img_names])
                self.poses = np.vstack((self.poses, poses))

        else:
            if train:
                if unsupervise:
                    split_filename = osp.join(base_dir, 'unsupervised_train_split.txt')
                else:
                    split_filename = osp.join(base_dir, 'dataset_train.txt')
            else:
                split_filename = osp.join(base_dir, 'dataset_test.txt')
            with open(split_filename, 'r') as f:
                data = f.readlines()
                self.imgs = [
                    osp.join(
                        base_dir,
                        l.split(' ')[0]
                    ) for l in data[3:] if not l.startswith('#')
                ]
                ps = np.asarray([
                    [float(num) for num in l.split(' ')[1:]]
                    for l in data[3:] if not l.startswith('#')
                ], dtype=np.float32)

            self.poses = np.zeros((ps.shape[0], 6))
            self.poses[:, :3] = ps[:, :3]
            self.poses[:, 3:] = np_qlog_t(ps[:, 3:])

        self.mask_sampling = self.config.mask_sampling
        if self.mask_sampling:
            muimg = read_grayscale_image(osp.join(base_dir, self.config.mu_mask_name))
            self.muimg = torch.tensor(muimg.transpose(2, 0, 1)).type(torch.FloatTensor)
            self.sigmaimg = self.muimg * (1 - self.muimg)

        pose_stats_filename = osp.join(data_dir, 'pose_stats.txt')
        if train and not real:
            mean_t = np.mean(self.poses[:, :3], axis=0)
            std_t = np.std(self.poses[:, :3], axis=0)
            np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')
            print("Saved")
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename)

        self.poses[:, :3] -= mean_t
        self.poses[:, :3] /= std_t

        # convert the pose to translation + log quaternion, align, normalize
        self.gt_idx = np.asarray(range(len(self.poses)))

        # camera model and image loader
        self.im_loader = partial(load_image)

    def __getitem__(self, index):
        if self.skip_images:
            img = None
            pose = self.poses[index]
        else:
            img = None
            while img is None:
                if self.undistort:
                    img = utils.load_image(self.imgs[index], loader=self.im_loader)
                else:
                    img = utils.load_image(self.imgs[index])
                pose = self.poses[index]
                index += 1
            index -= 1

        if self.target_transform is not None:
            pose = self.target_transform(pose)

        if self.skip_images:
            return img, pose

        if self.transform is not None:
            img = self.transform(img)
            if self.mask_sampling:
                mask = torch.normal(mean=self.muimg, std=self.sigmaimg) < self.config.sampling_threshold
                img = img * mask.type(torch.FloatTensor)

        return img, pose

    def __len__(self):
        return len(self.poses)

def main():
    from common.vis_utils import show_batch
    from torchvision.utils import make_grid
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    from mapnet.config import MapNetConfigurator
    from common.utils import get_configuration
    scene = 'KingsCollege'
    num_workers = 4
    transform = transforms.Compose([
        transforms.Scale(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor()])

    config = get_configuration(MapNetConfigurator())
    data_path = "/home/drinkingcoder/Dataset/Cambridge/"
    dset = Cambridge(scene, data_path, train=True, real=False,
                    transform=transform, data_dir=osp.join("..", "data", "Cambridge", "KingsCollege"),
                    config=config)
    print 'Loaded RobotCar scene {:s}, length = {:d}'.format(scene, len(dset))

    # plot the poses
    plt.figure()
    plt.scatter(dset.poses[:, 0], dset.poses[:, 1])
    plt.show()
    plt.figure()
    plt.scatter(dset.poses[:, 3], dset.poses[:, 5])
    plt.show()

    print(len(dset))
    data_loader = data.DataLoader(dset, batch_size=10, shuffle=True,
                                  num_workers=num_workers)

    batch_count = 0
    N = 2
    for batch in data_loader:
        print 'Minibatch {:d}'.format(batch_count)
        show_batch(make_grid(batch[0], nrow=5, padding=25, normalize=True))

        batch_count += 1
        if batch_count >= N:
            break

if __name__ == '__main__':
    main()
