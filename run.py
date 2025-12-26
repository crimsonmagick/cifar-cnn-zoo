import math

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch import cuda, max
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets, transforms
from torchvision.models import VGG16_Weights


def evaluate(model, loader, criterion, device):
    with torch.no_grad():
        pass


def main():
    data_dir = './data'
    batch_size = 32

    vgg16 = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True,
                                     download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False,
                                    download=True, transform=transform)

    val_size = math.floor(0.90 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Freeze all the layers of the pretrained model
    for param in vgg16.parameters():
        param.requires_grad = False

    # Replace the last fully - connected layer
    num_ftrs = vgg16.classifier[6].in_features
    vgg16.classifier[6] = nn.Linear(num_ftrs, 10)  # For CIFAR10 which has 10 classes

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(vgg16.classifier[6].parameters(), lr=0.001, momentum=0.9)

    # Training loop
    device = torch.device("cuda:0" if cuda.is_available() else "cpu")
    vgg16 = vgg16.to(device)

    for epoch in range(10):
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

            _, predicted = max(outputs, 1)  # class with highest logit
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    main()