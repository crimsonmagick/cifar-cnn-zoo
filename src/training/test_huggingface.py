import argparse

import torch
import torch.nn as nn

import logging

from evaluation import evaluate
from src.zoo import load_model, Arch, Cifar, get_test_loader

logger = logging.getLogger()


def main():
    parser = argparse.ArgumentParser(
        prog='CNN Huggingface Tester',
        description='Evaluates a CNN Using a Test Dataset, Loaded From Huggingface Safetensors'
    )
    parser.add_argument('model_name')
    args = parser.parse_args()
    model_name = args.model_name
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    arch_name, dataset_name = model_name.split("_")
    arch = Arch[arch_name.upper()]
    dataset = Cifar[dataset_name.upper()]
    model = load_model(arch, dataset, device)
    test_loader = get_test_loader(model.dataset)

    print(f"Testing {model.model_name}")
    criterion = nn.CrossEntropyLoss()
    evaluate(model, test_loader, criterion, device, prefix='Test')


if __name__ == '__main__':
    main()
