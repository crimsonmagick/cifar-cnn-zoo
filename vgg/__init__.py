from torch import optim
from torch.hub import load_state_dict_from_url
from torchvision import models
from torchvision.models import VGG16_Weights, VGG19_Weights, VGG11_Weights, VGG13_Weights, WeightsEnum

from vgg.modeling_cifar10 import VGGForCIFAR10

def _init_optimizer(model: VGGForCIFAR10):
    return optim.SGD(
        [
            {"params": model.model.features.parameters(), "lr": 1e-4},
            {"params": model.model.classifier.parameters(), "lr": 1e-3},
        ],
        momentum=0.9,
        weight_decay=5e-4,
    )

def vgg11_cifar10(transfer_learn=False):
    weights = VGG11_Weights.IMAGENET1K_V1 if transfer_learn else None
    vgg = models.vgg11(weights=weights)
    model = VGGForCIFAR10(vgg, "vgg11_cifar10")
    return model, _init_optimizer(model)


def vgg13_cifar10(transfer_learn=False):
    weights = VGG13_Weights.IMAGENET1K_V1 if transfer_learn else None
    vgg = models.vgg13(weights=weights)
    model = VGGForCIFAR10(vgg, "vgg13_cifar10")
    return model, _init_optimizer(model)


def vgg16_cifar10(transfer_learn=False):
    weights = VGG13_Weights.IMAGENET1K_V1 if transfer_learn else None
    vgg = models.vgg13(weights=weights)
    model = VGGForCIFAR10(vgg, "vgg16_cifar10")
    return model, _init_optimizer(model)


def vgg19_cifar10(transfer_learn=False):
    weights = VGG13_Weights.IMAGENET1K_V1 if transfer_learn else None
    vgg = models.vgg13(weights=weights)
    model = VGGForCIFAR10(vgg, "vgg19_cifar10")
    return model, _init_optimizer(model)
