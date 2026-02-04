import math
from enum import Enum

import torch
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms

_data_dir = './data'
_batch_size = 64
_seed = 12


class CIFAR(Enum):
    CIFAR10 = 10
    CIFAR100 = 100

    def dataset(self):
        if self is CIFAR.CIFAR10:
            return datasets.CIFAR10
        return datasets.CIFAR100


def _eval_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def _loader_kwargs():
    return {
        'batch_size': _batch_size,
        'num_workers': 16,
        'pin_memory': torch.cuda.is_available(),
        'persistent_workers': True,
        'prefetch_factor': 4,
        'drop_last': False
    }


def get_test_loader(cifar: CIFAR):
    eval_transform = _eval_transform()
    test_dataset = cifar.dataset()(root=_data_dir, train=False,
                                   download=True, transform=eval_transform)

    loader_kwargs = _loader_kwargs()
    return DataLoader(test_dataset, shuffle=False, **loader_kwargs)


def get_loaders(cifar: CIFAR):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            224,
            scale=(0.8, 1.0),  # zoom range
            ratio=(0.9, 1.1)  # aspect ratio jitter
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    eval_transform = _eval_transform()

    train_dataset = cifar.dataset()(root=_data_dir, train=True,
                                    download=True, transform=train_transform)
    val_dataset = cifar.dataset()(root=_data_dir, train=True,
                                  download=True, transform=eval_transform)
    test_dataset = cifar.dataset()(root=_data_dir, train=False,
                                   download=True, transform=eval_transform)

    val_size = math.floor(0.10 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, _ = random_split(train_dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(_seed))
    _, val_dataset = random_split(val_dataset, [train_size, val_size],
                                  generator=torch.Generator().manual_seed(_seed))
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = eval_transform

    loader_kwargs = _loader_kwargs()
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader
