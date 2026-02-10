import argparse

import torch
import torch.nn as nn

import logging

from evaluation import evaluate
from src.crimson_magick.cifar_zoo.model_services import model_for_testing_safetensor

logger = logging.getLogger()


def main():
    parser = argparse.ArgumentParser(
        prog='CNN Safetensor Tester',
        description='Evaluates a CNN Using a Test Dataset, Loaded From Safetensors'
    )
    parser.add_argument('model_path')
    args = parser.parse_args()
    model_name = args.model_path
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, test_loader = model_for_testing_safetensor(model_name, device)

    print(f"Testing {model.model_name}")

    criterion = nn.CrossEntropyLoss()
    evaluate(model, test_loader, criterion, device, prefix='Test')


if __name__ == '__main__':
    main()
