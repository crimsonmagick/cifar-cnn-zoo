import math
from enum import Enum

import torch
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms

_data_dir = './data'


class CIFAR(Enum):
    CIFAR10 = 10
    CIFAR100 = 100

    def dataset(self):
        if self is CIFAR.CIFAR10:
            return datasets.CIFAR10
        return datasets.CIFAR100


def get_loaders(cifar: CIFAR):
    batch_size = 64
    seed = 12
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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

    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': 16,
        'pin_memory': torch.cuda.is_available(),
        'persistent_workers': True,
        'prefetch_factor': 4,
        'drop_last': False
    }

    train_dataset = cifar.dataset()(root=_data_dir, train=True,
                                    download=True, transform=train_transform)
    val_dataset = cifar.dataset()(root=_data_dir, train=True,
                                  download=True, transform=eval_transform)
    train_eval_dataset = cifar.dataset()(root=_data_dir, train=True,
                                         download=True, transform=eval_transform)
    test_dataset = cifar.dataset()(root=_data_dir, train=False,
                                   download=True, transform=eval_transform)

    val_size = math.floor(0.10 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, _ = random_split(train_dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(seed))
    _, val_dataset = random_split(val_dataset, [train_size, val_size],
                                  generator=torch.Generator().manual_seed(seed))
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = eval_transform

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    # train_eval_loader = DataLoader(train_eval_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader
