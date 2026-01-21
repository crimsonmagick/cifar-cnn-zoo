import argparse
import sys
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets, transforms

import logging

from evaluation import evaluate
from resnet import resnet18_cifar10
from vgg import vgg16_cifar10, vgg19_cifar10, vgg11_cifar10, vgg13_cifar10

logger = logging.getLogger()


def main():
    parser = argparse.ArgumentParser(
        prog='CNN Tester',
        description='Evaluates a CNN Using a Test Dataset'
    )
    parser.add_argument('checkpoint')
    args = parser.parse_args()
    checkpoint_path = args.checkpoint

    data_dir = './data'
    batch_size = 64

    model, _ = resnet18_cifar10(transfer_learn=False)

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': 16,
        'pin_memory': torch.cuda.is_available(),
        'persistent_workers': True,
        'prefetch_factor': 4,
        'drop_last': False
    }

    test_dataset = datasets.CIFAR10(root=data_dir, train=False,
                                    download=True, transform=eval_transform)

    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    for param in model.parameters():
        param.requires_grad = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    epoch_count = 0

    try:
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        epoch_count = checkpoint['epoch'] + 1
    except FileNotFoundError as e:
        logger.error(f"Checkpoint {checkpoint_path} was not found, exception={repr(e)} Exiting...")
        sys.exit()
    except Exception as e:
        logger.error(
            f"Encountered exception while loading checkpoint {checkpoint_path}, exception={repr(e)}. Exiting...")
        sys.exit()

    print(f"Testing {model.model_name}")

    criterion = nn.CrossEntropyLoss()
    evaluate(model, test_loader, criterion, device, prefix='Test')
    print(f"Trained for {epoch_count} epochs")

if __name__ == '__main__':
    main()
