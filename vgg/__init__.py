from torch.hub import load_state_dict_from_url
from torchvision import models
from torchvision.models import VGG16_Weights, VGG19_Weights, VGG11_Weights, VGG13_Weights, WeightsEnum

from vgg.modeling_cifar10 import VGGForCIFAR10


def vgg11_cifar10(transfer_learn=False):
    weights = VGG11_Weights.IMAGENET1K_V1 if transfer_learn else None
    vgg11 = models.vgg11(weights=weights)
    return VGGForCIFAR10(vgg11, "vgg11_cifar10")


def vgg13_cifar10(transfer_learn=False):
    def get_state_dict(self, *args, **kwargs):
        kwargs.pop("check_hash")
        return load_state_dict_from_url(self.url, *args, **kwargs)
    WeightsEnum.get_state_dict = get_state_dict
    weights = VGG13_Weights.IMAGENET1K_V1 if transfer_learn else None
    vgg13 = models.vgg13(weights=weights)
    return VGGForCIFAR10(vgg13, "vgg13_cifar10")


def vgg16_cifar10(transfer_learn=False):
    weights = VGG16_Weights.IMAGENET1K_V1 if transfer_learn else None
    vgg16 = models.vgg16(weights=weights)
    return VGGForCIFAR10(vgg16, "vgg16_cifar10")


def vgg19_cifar10(transfer_learn=False):
    weights = VGG19_Weights.IMAGENET1K_V1 if transfer_learn else None
    vgg19 = models.vgg19(weights=weights)
    return VGGForCIFAR10(vgg19, "vgg19_cifar10")
