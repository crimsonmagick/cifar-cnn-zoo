from torchvision import models
from torchvision.models import VGG16_Weights, VGG19_Weights, VGG11_Weights

from vgg.modeling_cifar10 import VGGForCIFAR10


def vgg11_cifar10(transfer_learn=False):
    weights = VGG11_Weights.IMAGENET1K_V1 if transfer_learn else None
    vgg16 = models.vgg11(weights=weights)
    return VGGForCIFAR10(vgg16, "vgg11_cifar10")


def vgg16_cifar10(transfer_learn=False):
    weights = VGG16_Weights.IMAGENET1K_V1 if transfer_learn else None
    vgg16 = models.vgg16(weights=weights)
    return VGGForCIFAR10(vgg16, "vgg16_cifar10")


def vgg19_cifar10(transfer_learn=False):
    weights = VGG19_Weights.IMAGENET1K_V1 if transfer_learn else None
    vgg19 = models.vgg19(weights=weights)
    return VGGForCIFAR10(vgg19, "vgg19_cifar10")
