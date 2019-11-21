import os.path as osp

from dataset_loader.seven_scenes import SevenScenes
from dataset_loader.robotcar import RobotCar
from dataset_loader.sensetime import SenseTime
from dataset_loader.cambridge import Cambridge
from dataset_loader.composite import MF, MaskMF
from common.utils import safe_collate
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np

def get_rnn_test_dataloader(config, train=False):
    data_dir = osp.join('..', 'data', config.dataset)
    stats_filename = osp.join(data_dir, config.scene, 'stats.txt')
    stats = np.loadtxt(stats_filename)

    # transformer
    data_transform = transforms.Compose(
        [
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=stats[0],
                std=np.sqrt(stats[1])
            ),
        ]
    )
    target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

    #dataset
    kwargs = dict(
        scene=config.scene,
        data_path=config.dataset_path,
        transform=data_transform,
        target_transform=target_transform,
        seed=config.seed,
        no_duplicates=True,
        data_dir=config.preprocessed_data_path,
        dataset=config.dataset,
        skip=config.skip,
        steps=config.steps,
        variable_skip=False,
        config=config
    )
    test_dataset = MF(train=False, real=False, **kwargs)

    # dataloader
    dataloader_kwargs = dict(
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=safe_collate
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        **dataloader_kwargs
    )
    return test_dataloader


def get_test_dataloader(config, train=False):
    data_dir = osp.join('..', 'data', config.dataset)
    stats_filename = osp.join(data_dir, config.scene, 'stats.txt')
    stats = np.loadtxt(stats_filename)
    # transformer
    data_transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1]))])
    target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())


    kwargs = dict(
        scene=config.scene,
        data_path=config.dataset_path,
        train=train,
        transform=data_transform,
        target_transform=target_transform,
        seed=config.seed,
        data_dir=config.preprocessed_data_path,
        config=config
    )
    if config.dataset == '7Scenes':
        dataset = SevenScenes(**kwargs)
    elif config.dataset == 'RobotCar':
        dataset = RobotCar(**kwargs)
    elif config.dataset == 'SenseTime':
        dataset = SenseTime(**kwargs)
    elif config.dataset == 'Cambridge' or config.dataset == 'NewCambridge':
        dataset = Cambridge(**kwargs)

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    return loader

def get_train_transforms(config):
    print("Load data...")
    stats_file = osp.join(config.preprocessed_data_path, 'stats.txt')
    stats = np.loadtxt(stats_file)

    if config.color_jitter > 0:
        assert config.color_jitter <= 1.0
        print("Using color jitter data augementation")
        color_jitter_transform = transforms.ColorJitter(
            brightness=config.color_jitter,
            contrast=config.color_jitter,
            saturation=config.color_jitter,
            hue=config.color_jitter/2
        )
        data_transform = transforms.Compose(
            [
                transforms.Resize(config.image_size),
                color_jitter_transform,
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=stats[0],
                    std=np.sqrt(stats[1])
                ),
            ]
        )
    else:
        data_transform = transforms.Compose(
            [
                transforms.Resize(config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=stats[0],
                    std=np.sqrt(stats[1])
                ),
            ]
        )

    target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

    return data_transform, target_transform

def get_mask_transforms(configuration):
    mask_transform = transforms.Compose(
        [
            transforms.Resize(configuration.mask_size),
            transforms.ToTensor()
            # transforms.Lambda(lambda x: x/255)
        ]
    )

    return mask_transform

def get_mapnet_train_dataloader(config):
    data_transform, target_transform = get_train_transforms(config)
    kwargs = dict(
        scene=config.scene,
        data_path=config.dataset_path,
        transform=data_transform,
        target_transform=target_transform,
        seed=config.seed,
        data_dir=config.preprocessed_data_path,
        dataset=config.dataset,
        skip=config.skip,
        steps=config.steps,
        variable_skip=False,
        config=config
    )
    train_dataset = MF(train=True, real=False, **kwargs)
    valid_dataset = MF(train=False, real=False, **kwargs)

    # dataloader
    dataloader_kwargs = dict(
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=safe_collate
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        **dataloader_kwargs
    )
    if config.do_val:
        val_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            **dataloader_kwargs
        )
    else:
        val_dataloader = None

    return train_dataloader, val_dataloader

def get_posenet_train_dataloader(config):
    data_transform, target_transform = get_train_transforms(config)
    kwargs = dict(
        scene=config.scene,
        data_path=config.dataset_path,
        transform=data_transform,
        target_transform=target_transform,
        seed=config.seed,
        data_dir=config.preprocessed_data_path,
        config=config
    )
    if config.dataset == '7Scenes':
        train_data = SevenScenes(train=True,  **kwargs)
        valid_data = SevenScenes(train=False,  **kwargs)
    elif config.dataset == 'RobotCar':
        train_data = RobotCar(train=True,  **kwargs)
        valid_data = RobotCar(train=False,  **kwargs)
    elif config.dataset == 'SenseTime':
        train_data = SenseTime(train=True, **kwargs)
        valid_data = SenseTime(train=False, **kwargs)
    elif config.dataset == 'Cambridge' or config.dataset == 'NewCambridge':
        train_data = Cambridge(train=True, **kwargs)
        valid_data = Cambridge(train=False, **kwargs)
    else:
        raise NotImplementedError

    dataloader_kwargs = dict(
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=safe_collate
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        **dataloader_kwargs
    )
    if config.do_val:
        val_dataloader = torch.utils.data.DataLoader(
            valid_data,
            **dataloader_kwargs
        )
    else:
        val_dataloader = None

    return train_dataloader, val_dataloader

def get_maskmapnet_train_dataloader(config):
    data_transform, target_transform = get_train_transforms(config)
    mask_transform = get_mask_transforms(config)
    kwargs = dict(
        scene=config.scene,
        data_path=config.dataset_path,
        transform=data_transform,
        target_transform=target_transform,
        seed=config.seed,
        data_dir=config.preprocessed_data_path,
        dataset=config.dataset,
        skip=config.skip,
        steps=config.steps,
        variable_skip=False,
        config=config
    )
    maskkwargs = dict(
        scene=config.scene,
        data_path=config.dataset_path,
        transform=data_transform,
        target_transform=target_transform,
        mask_transform=mask_transform,
        seed=config.seed,
        data_dir=config.preprocessed_data_path,
        dataset=config.dataset,
        skip=config.skip,
        steps=config.steps,
        variable_skip=False
    )
    train_data = MaskMF(train=True, real=False, **maskkwargs)
    valid_data = MF(train=False, real=False, **kwargs)

    # dataloader
    dataloader_kwargs = dict(
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=safe_collate
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        **dataloader_kwargs
    )
    if config.do_val:
        val_dataloader = torch.utils.data.DataLoader(
            valid_data,
            **dataloader_kwargs
        )
    else:
        val_dataloader = None

    return train_dataloader, val_dataloader

