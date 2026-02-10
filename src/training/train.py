import argparse
import os
import sys
from datetime import datetime, timezone

import torch
import torch.nn as nn
import logging

from constants import CHECKPOINT_DIR
from evaluation import evaluate
from crimson_magick.cifar_zoo.model_services import model_for_training

logger = logging.getLogger()


def save_model(accuracy, acc_type, model, optimizer, epoch):
    checkpoint_path = f'{CHECKPOINT_DIR}/{model.model_name}'
    os.makedirs(checkpoint_path, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    accuracy_str = "{:.2f}".format(accuracy).replace(".", "_")
    base_model_name, dataset_name = model.model_name.split("_")
    torch.save({
        "base_model_name": base_model_name,
        "dataset_name": dataset_name,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch
    }, f"{checkpoint_path}/{acc_type}_{accuracy_str}_{timestamp}.pt")


def main():
    parser = argparse.ArgumentParser(
        prog='CNN Trainer',
        description='Updates and Trains CNNs for CIFAR10 and CIFAR100'
    )
    parser.add_argument('model_name')
    parser.add_argument('dataset_name')
    parser.add_argument('iterations', type=int)
    parser.add_argument('-c', '--checkpoint')
    args = parser.parse_args()
    model_name = args.model_name
    dataset_name = args.dataset_name
    iterations = args.iterations
    checkpoint_path = args.checkpoint
    train(model_name, dataset_name, iterations, checkpoint_path)


def train(model_name, dataset_name, iterations, checkpoint_path):
    model, optimizer, (train_loader, val_loader, test_loader) = model_for_training(model_name, dataset_name, True)

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
    epoch = 0
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


    _, test_accuracy = evaluate(model, test_loader, criterion, device, prefix='Test')
    if epoch >= 0:
        save_model(test_accuracy, "test", model, optimizer, epoch)


if __name__ == '__main__':
    main()
