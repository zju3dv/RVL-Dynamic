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
from robotcar_sdk.interpolate_poses import interpolate_vo_poses,\
  interpolate_ins_poses
from robotcar_sdk.camera_model import CameraModel
from robotcar_sdk.image import load_image
import common.utils as utils
from functools import partial
from common.pose_utils import process_poses
import pickle
import cv2
from common.img_utils import read_grayscale_image
from scipy.stats import norm

class RobotCar(data.Dataset):
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

    # decide which sequences to use
    if train:
      if unsupervise:
        split_filename = osp.join(base_dir, 'unsupervised_train_split.txt')
      else:
        split_filename = osp.join(base_dir, 'train_split.txt')
    else:
      split_filename = osp.join(base_dir, 'test_split.txt')
    with open(split_filename, 'r') as f:
      seqs = [l.rstrip() for l in f if not l.startswith('#')]

    print("Extract data from {}".format(seqs))

    ps = {}
    ts = {}
    vo_stats = {}
    self.imgs = []
    self.masks = []
    self.mask_sampling = self.config.mask_sampling
    self.uniform_sampling = self.config.uniform_sampling
    #if 'mask_image' in self.config:
    #    self.mask_image = self.config.mask_image
    #else:
    #    self.mask_image = False
    self.mask_image = False

    if self.mask_sampling:
        muimg = read_grayscale_image(osp.join(base_dir, self.config.mu_mask_name))
        self.muimg = torch.tensor(muimg.transpose(2, 0, 1)).type(torch.FloatTensor)
        if self.uniform_sampling:
            self.muimg.fill_(self.muimg.mean())
        self.sigmaimg = self.muimg * (1 - self.muimg)
        if not self.train:
          self.prob_mask = torch.FloatTensor(norm.cdf((self.config.sampling_threshold - self.muimg) / self.sigmaimg))
    for seq in seqs:
      seq_dir = osp.join(base_dir, seq)
      seq_data_dir = osp.join(data_dir, seq)

      # read the image timestamps
      ts_filename = osp.join(seq_dir, 'stereo.timestamps')
      with open(ts_filename, 'r') as f:
        ts[seq] = [int(l.rstrip().split(' ')[0]) for l in f]

      if real:  # poses from integration of VOs
        if vo_lib == 'stereo':
          vo_filename = osp.join(seq_dir, 'vo', 'vo.csv')
          p = np.asarray(interpolate_vo_poses(vo_filename, ts[seq], ts[seq][0]))
        elif vo_lib == 'gps':
          vo_filename = osp.join(seq_dir, 'gps', 'gps_ins.csv')
          p = np.asarray(interpolate_ins_poses(vo_filename, ts[seq], ts[seq][0]))
        else:
          raise NotImplementedError
        vo_stats_filename = osp.join(seq_data_dir, '{:s}_vo_stats.pkl'.
                                     format(vo_lib))
        with open(vo_stats_filename, 'r') as f:
          vo_stats[seq] = pickle.load(f)
        ps[seq] = np.reshape(p[:, :3, :], (len(p), -1))
      else:  # GT poses
        pose_filename = osp.join(seq_dir, 'gps', 'ins.csv')
        p = np.asarray(interpolate_ins_poses(pose_filename, ts[seq], ts[seq][0]))
        ps[seq] = np.reshape(p[:, :3, :], (len(p), -1))
        vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

      self.imgs.extend([osp.join(seq_dir, 'stereo', 'centre', '{:d}.png'.
                                 format(t)) for t in ts[seq]])
      if self.mask_image:
          self.masks.extend([osp.join(seq_dir, 'mask', '{:d}.png'.
                                     format(t)) for t in ts[seq]])

    # read / save pose normalization information
    poses = np.empty((0, 12))
    for p in ps.values():
      poses = np.vstack((poses, p))
    pose_stats_filename = osp.join(data_dir, 'pose_stats.txt')
    if train and not real:
      mean_t = np.mean(poses[:, [3, 7, 11]], axis=0)
      std_t = np.std(poses[:, [3, 7, 11]], axis=0)
      np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')
    else:
      mean_t, std_t = np.loadtxt(pose_stats_filename)

    # convert the pose to translation + log quaternion, align, normalize
    self.poses = np.empty((0, 6))
    for seq in seqs:
      pss = process_poses(poses_in=ps[seq], mean_t=mean_t, std_t=std_t,
                          align_R=vo_stats[seq]['R'], align_t=vo_stats[seq]['t'],
                          align_s=vo_stats[seq]['s'])
      self.poses = np.vstack((self.poses, pss))
    self.gt_idx = np.asarray(range(len(self.poses)))

    # camera model and image loader
    camera_model = CameraModel(osp.join('..', 'data', 'robotcar_camera_models'),
                               osp.join('stereo', 'centre'))
    self.im_loader = partial(load_image, model=camera_model)

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
          if self.mask_image:
	    #print(self.masks[index])
            maskimg = cv2.imread(self.masks[index])
            mask = torch.tensor((maskimg.transpose(2, 0, 1) < 50).astype(int)).type(torch.FloatTensor)

        pose = self.poses[index]
        index += 1
      index -= 1

    if self.target_transform is not None:
      pose = self.target_transform(pose)

    if self.skip_images:
      return img, pose

    if self.transform is not None:
      img = self.transform(img)
      if self.mask_image:
        img = img * mask
      if self.mask_sampling:
        # if self.train:
        mask = torch.normal(mean=self.muimg, std=self.sigmaimg) < self.config.sampling_threshold
        img = img * mask.type(torch.FloatTensor)
        # else:
        #   img = img * self.prob_mask

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
  scene = 'full'
  num_workers = 4
  transform = transforms.Compose([
    transforms.Scale(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor()])

  config = get_configuration(MapNetConfigurator())
  config.uniform_sampling = False
  config.mask_sampling = False
  config.mask_image = True

  data_path = "/home/drinkingcoder/Dataset/robotcar/"
  dset = RobotCar(scene, data_path, train=True, real=False,
          transform=transform, data_dir=osp.join("..", "data", "RobotCar"),
          config=config)
  print 'Loaded RobotCar scene {:s}, length = {:d}'.format(scene, len(dset))

  # plot the poses
  plt.figure()
  plt.plot(dset.poses[:, 0], dset.poses[:, 1])
  plt.show()

  print(len(dset))
  data_loader = data.DataLoader(dset, batch_size=10, shuffle=True,
                                num_workers=num_workers)
  #plt.imshow(dset.muimg.data.numpy()[0, :, :])
  #plt.imshow(dset[0].data.numpy())

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
