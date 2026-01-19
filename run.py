import argparse
import math
import sys
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets, transforms
from torchvision.models import VGG16_Weights

import logging

logger = logging.getLogger()


def evaluate(model, loader, criterion, device, *, prefix='Val'):
    with torch.no_grad():
        total = 0
        correct = 0
        running_loss = 0
        was_training = model.training

        model.eval()
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = 100.0 * correct / total
        if was_training:
            model.train()

        avg_loss = running_loss / len(loader)
        print(f'{prefix}: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        return avg_loss, accuracy


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

    data_dir = './data'
    batch_size = 64
    seed = 12

    vgg16 = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

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

    train_dataset = datasets.CIFAR10(root=data_dir, train=True,
                                     download=True, transform=None)
    train_eval_dataset = datasets.CIFAR10(root=data_dir, train=True,
                                          download=True, transform=eval_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False,
                                    download=True, transform=eval_transform)

    val_size = math.floor(0.10 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(seed))
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = eval_transform

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    for param in vgg16.parameters():
        param.requires_grad = True

    # Replace the last fully - connected layer
    num_ftrs = vgg16.classifier[6].in_features
    vgg16.classifier[6] = nn.Linear(num_ftrs, 10)  # For CIFAR10 which has 10 classes

    # train classifier and final vgg layer
    # for param in vgg16.classifier:
    #     param.requires_grad = True
    # for param in vgg16.features[24:].parameters():
    #     param.requires_grad = True

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        [
            # {"params": vgg16.features[24:].parameters(), "lr": 1e-4},
            {"params": vgg16.features.parameters(), "lr": 1e-4},
            {"params": vgg16.classifier[6].parameters(), "lr": 1e-3},
        ],
        momentum=0.9,
        weight_decay=5e-4,
    )

    # Training loop
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vgg16 = vgg16.to(device)

    initial_epoch = 0

    # optionally load
    if checkpoint_path:
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=device)
            vgg16.load_state_dict(checkpoint['model_state'])
            # optimizer.load_state_dict(checkpoint['optimizer_state'])
            initial_epoch = checkpoint['epoch'] + 1
        except FileNotFoundError as e:
            logger.error(f"Checkpoint {checkpoint_path} was not found, exception={repr(e)} Exiting...")
            sys.exit()
        except Exception as e:
            logger.error(
                f"Encountered exception while loading checkpoint {checkpoint_path}, exception={repr(e)}. Exiting...")
            sys.exit()

    for epoch in range(initial_epoch, initial_epoch + iterations):
        vgg16.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = vgg16(inputs)
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
        evaluate(vgg16, val_loader, criterion, device)
        # evaluate(vgg16, train_eval_loader, criterion, device, prefix="Train(Eval)")

    evaluate(vgg16, test_loader, criterion, device, prefix='Test')
    if epoch >= 0:
        timestamp = datetime.now(timezone.utc).isoformat()
        torch.save({
            "model_state": vgg16.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch
        }, f"models/vgg16_cifar10_checkpoint{timestamp}.pth")


if __name__ == '__main__':
    main()
