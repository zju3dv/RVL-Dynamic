
# coding: utf-8

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
from dataset_loader.cambridge import Cambridge
from easydict import EasyDict

image_size = (256, 480)
def safe_collate(batch):
    batch = filter(lambda x: x is not None, batch)
    return default_collate(batch)

parser = argparse.ArgumentParser(description='Preprocess data')
parser.add_argument('--data_dir', type=str, default='/home/drinkingcoder/Dataset/Cambridge/')
parser.add_argument('--scene', type=str, default='KingsCollege')
parser.add_argument('--train', action='store_true')
args = parser.parse_args()

batch_size = 8
num_workers = batch_size
from PIL import Image

config = EasyDict(
    {
        'mask_sampling': False,
        'data_dir': osp.join('..', 'data', 'Cambridge', args.scene)
    }
)

if not osp.isdir(config.data_dir):
    os.mkdir(config.data_dir)

def ResizeImages(train=True):
    data_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Lambda(lambda x:np.asarray(x))
    ])
    dset = Cambridge(
        scene=args.scene,
        data_path=args.data_dir,
        train=train,
        transform=data_transform,
        config=config,
        data_dir=config.data_dir
    )
    loader = DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=safe_collate,
        shuffle=False
    )
    base_dir = osp.join(args.data_dir, args.scene)
    if train:
            split_filename = osp.join(base_dir, 'dataset_train.txt')
    else:
        split_filename = osp.join(base_dir, 'dataset_test.txt')
    with open(split_filename, 'r') as f:
        data = f.readlines()
        im_fns = [
            osp.join(
                base_dir,
                l.split(' ')[0]
            ) for l in data[3:] if not l.startswith('#')
        ]

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


ResizeImages(args.train)

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
    transform=data_transform,
    config=config,
    data_dir=config.data_dir
)
dset = Cambridge(**kwargs)

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

output_fn = osp.join(config.data_dir, 'stats.txt')
np.savetxt(output_fn, np.vstack((mean_p, std_p)), fmt='%8.7f')
print('{:s} written'.format(output_fn))

