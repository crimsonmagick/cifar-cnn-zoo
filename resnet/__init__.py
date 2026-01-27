from torch import optim
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

from fine_tuned.datasets import get_loaders, CIFAR
from resnet.modeling_cifar import ResNetForCIFAR


def _init_optimizer(model):
    base_params = [p for n, p in model.model.named_parameters() if not n.startswith("fc")]
    head_params = model.model.fc.parameters()
    return optim.SGD(
        [
            {"params": base_params, "lr": 1e-4},
            {"params": head_params, "lr": 1e-3},
        ],
        momentum=0.9,
        weight_decay=5e-4,
    )

def resnet18_cifar10(transfer_learn=False):
    weights = ResNet18_Weights.IMAGENET1K_V1 if transfer_learn else None
    resnet = models.resnet18(weights=weights)
    model = ResNetForCIFAR(resnet, "resnet18_cifar10", CIFAR.CIFAR10)
    return model, _init_optimizer(model), get_loaders(CIFAR.CIFAR10)

def resnet34_cifar10(transfer_learn=False):
    weights = ResNet34_Weights.IMAGENET1K_V1 if transfer_learn else None
    resnet = models.resnet34(weights=weights)
    model = ResNetForCIFAR(resnet, "resnet34_cifar10", CIFAR.CIFAR10)
    optimizer = _init_optimizer(model)
    return model, _init_optimizer(model), get_loaders(CIFAR.CIFAR10)

def resnet50_cifar100(transfer_learn=False):
    weights = ResNet50_Weights.IMAGENET1K_V1 if transfer_learn else None
    resnet = models.resnet50(weights=weights)
    model = ResNetForCIFAR(resnet, "renet50_cifar100", CIFAR.CIFAR100)
    return model, _init_optimizer(model), get_loaders(CIFAR.CIFAR100)
