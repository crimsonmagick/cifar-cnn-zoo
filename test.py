import argparse
import sys

import torch
import torch.nn as nn


import logging

import mobilenet
from evaluation import evaluate

logger = logging.getLogger()


def main():
    parser = argparse.ArgumentParser(
        prog='CNN Tester',
        description='Evaluates a CNN Using a Test Dataset'
    )
    parser.add_argument('checkpoint')
    args = parser.parse_args()
    checkpoint_path = args.checkpoint

    model, _, (_, _, test_loader) = mobilenet.mobilenet_v2_cifar100(transfer_learn=False)

    for param in model.parameters():
        param.requires_grad = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    epoch_count = 0

    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
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
