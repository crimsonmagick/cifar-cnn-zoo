import torch


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
