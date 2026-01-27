import argparse
import math
import sys
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets, transforms

import logging

import mobilenet
from evaluation import evaluate
from resnet import resnet18_cifar10, resnet34_cifar10, resnet50_cifar100
from vgg import vgg16_cifar10, vgg19_cifar10, vgg11_cifar10, vgg13_cifar10

logger = logging.getLogger()


def main():
    parser = argparse.ArgumentParser(
        prog='CNN Trainer',
        description='Updates and Trains CNNs for CIFAR10 and CIFAR100'
    )
    parser.add_argument('iterations', type=int)
    parser.add_argument('-c', '--checkpoint')
    args = parser.parse_args()
    iterations = args.iterations
    checkpoint_path = args.checkpoint

    model, optimizer, (train_loader, val_loader, test_loader) = mobilenet.mobilenet_v1_cifar100(transfer_learn=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    initial_epoch = 0

    # optionally load
    if checkpoint_path:
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            initial_epoch = checkpoint['epoch'] + 1
        except FileNotFoundError as e:
            logger.error(f"Checkpoint {checkpoint_path} was not found, exception={repr(e)} Exiting...")
            sys.exit()
        except Exception as e:
            logger.error(
                f"Encountered exception while loading checkpoint {checkpoint_path}, exception={repr(e)}. Exiting...")
            sys.exit()

    print(f"Beginning training loop for {model.model_name}")
    criterion = nn.CrossEntropyLoss()
    for epoch in range(initial_epoch, initial_epoch + iterations):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            predicted = outputs.argmax(dim=1)  # class with highest logit
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        print(f'Train: Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        evaluate(model, val_loader, criterion, device)

    evaluate(model, test_loader, criterion, device, prefix='Test')
    if epoch >= 0:
        timestamp = datetime.now(timezone.utc).isoformat()
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch
        }, f"models/{model.model_name}_checkpoint_{timestamp}.pth")


if __name__ == '__main__':
    main()
