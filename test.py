import argparse
import sys

import torch
import torch.nn as nn

import logging

from evaluation import evaluate
from model_services import model_for_training, model_from_checkpoint, model_for_testing

logger = logging.getLogger()


def main():
    parser = argparse.ArgumentParser(
        prog='CNN Tester',
        description='Evaluates a CNN Using a Test Dataset'
    )
    parser.add_argument('checkpoint')
    args = parser.parse_args()
    checkpoint_path = args.checkpoint
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, test_loader = model_for_testing(checkpoint_path, device)

    print(f"Testing {model.model_name}")

    criterion = nn.CrossEntropyLoss()
    evaluate(model, test_loader, criterion, device, prefix='Test')


if __name__ == '__main__':
    main()
