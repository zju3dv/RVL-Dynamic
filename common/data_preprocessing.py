
# coding: utf-8
"""
Copyright SenseTime.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

# In[1]:


# process configuration
import sys
sys.path.insert(0, '../')
import cv2
import argparse
import os
import os.path as osp

import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from dataset_loader.sensetime import SenseTime

image_size = (256, 563)
def safe_collate(batch):
    batch = filter(lambda x: x is not None, batch)
    return default_collate(batch)

# extract frames from video
def extract_frames(video_path, images_path):
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    while success:
        image_name = "{:s}/Frame{:06d}.jpg".format(images_path, count)
        cv2.imwrite(image_name, image)     # save frame as JPEG file
        success,image = vidcap.read()
        print("{:s} processed.".format(image_name))
        count += 1
    np.savetxt(osp.join(images_path, 'count.txt'), [count], fmt='%d')

parser = argparse.ArgumentParser(description='Preprocess data')
parser.add_argument('--data_dir', type=str, default='/home/drinkingcoder/Dataset/sensetime/')
parser.add_argument('--video', type=str, default='CAM101.avi')
parser.add_argument('--scene', type=str, default='Museum')
args = parser.parse_args()

process_list_path = osp.join(args.data_dir, args.scene, 'process_list.txt')
with open(process_list_path) as f:
    process_list = [line.rstrip('\n') for line in f]
print('process_list = {}'.format(process_list))


# In[2]:


# extract frames from videos of sequence in process_list

for seq in process_list:
    video_path = osp.join(args.data_dir, args.scene, seq, args.video)
    images_path = osp.join(args.data_dir, args.scene, seq, 'images')
    print(images_path)
    if not osp.isdir(images_path):
        os.makedirs(images_path)
    resized_images_fn = osp.join(args.data_dir, args.scene, seq, 'resized_images')
    print(resized_images_fn)
    if not osp.isdir(resized_images_fn):
        os.makedirs(resized_images_fn)

    extract_frames(video_path, images_path)


# In[3]:


batch_size = 8
num_workers = batch_size
from PIL import Image

def ResizeImages(train=True):
    data_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Lambda(lambda x:np.asarray(x))
    ])
    dset = SenseTime(scene=args.scene, data_path=args.data_dir,train=train, transform=data_transform)
    loader = DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=safe_collate
    )
    base_dir = osp.join(args.data_dir, args.scene)
    if train:
        split_filename = osp.join(base_dir, 'train_split.txt')
    else:
        split_filename = osp.join(base_dir, 'test_split.txt')
    with open(split_filename, 'r') as f:
        seqs = [l.rstrip() for l in f if not l.startswith('#')]
    im_fns = []
    for seq in seqs:
        seq_dir = osp.join(base_dir, seq)
        count = np.loadtxt(osp.join(seq_dir, 'images', 'count.txt'))
        print('count = {}'.format(count))
        im_fns.extend([osp.join(seq_dir, 'resized_images', 'Frame{:06d}.jpg'.format(idx)) for idx in range(count)])
        np.savetxt(osp.join(seq_dir, 'resized_images', 'count.txt'), [count], fmt='%d')
    assert len(dset) == len(im_fns)

    for batch_idx, (imgs, _) in enumerate(loader):
        for idx, im in enumerate(imgs):
            im_fn = im_fns[batch_idx * batch_size + idx]
            im = Image.fromarray(im.numpy())
            try:
                im.save(im_fn)
            except IOError:
                print('IOError while saving {:s}'.format(im_fn))
        if batch_idx % 50 == 0:
            print('Processed {:d}/{:d}'.format(batch_idx*batch_size, len(dset)))


# In[5]:


ResizeImages(False)


# In[6]:


# get statistical information from training set only

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor()
])
kwargs = dict(
    scene=args.scene,
    data_path=args.data_dir,
    train=True,
    transform=data_transform
)
dset = SenseTime(**kwargs)

loader = DataLoader(
    dset,
    batch_size=batch_size,
    num_workers=num_workers,
    collate_fn=safe_collate
)
acc = np.zeros((3, image_size[0], image_size[1]))
sq_acc = np.zeros((3, image_size[0], image_size[1]))
for batch_idx, (imgs, _) in enumerate(loader):
    imgs = imgs.numpy()
    acc += np.sum(imgs, axis=0)
    sq_acc += np.sum(imgs**2, axis=0)

    if batch_idx % 50 == 0:
        print('Accumulated {:d}/{:d}'.format(batch_idx * batch_size, len(dset)))



# In[7]:


N = len(dset) * acc.shape[1] * acc.shape[2]
mean_p = np.asarray([np.sum(acc[c]) for c in xrange(3)])
mean_p /= N
print('Mean pixel = {}'.format(mean_p))

std_p = np.asarray([np.sum(sq_acc[c]) for c in xrange(3)])
std_p /= N
std_p -= (mean_p ** 2)
print('Std pixel = {}'.format(std_p))

output_fn = osp.join(args.data_dir, args.scene, 'stats.txt')
np.savetxt(output_fn, np.vstack((mean_p, std_p)), fmt='%8.7f')
print('{:s} written'.format(output_fn))

