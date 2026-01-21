from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights

from resnet.modeling_cifar10 import ResNetForCIFAR10


def resnet18_cifar10(transfer_learn=False):
    weights = ResNet18_Weights.IMAGENET1K_V1 if transfer_learn else None
    resnet = models.resnet18(weights=weights)
    return ResNetForCIFAR10(resnet, "resnet18_cifar10")

def resnet34_cifar10(transfer_learn=False):
    weights = ResNet34_Weights.IMAGENET1K_V1 if transfer_learn else None
    resnet = models.resnet34(weights=weights)
    return ResNetForCIFAR10(resnet, "resnet34_cifar10")
