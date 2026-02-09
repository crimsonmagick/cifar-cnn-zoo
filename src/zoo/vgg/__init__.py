import logging

from torch import optim
from torchvision import models
from torchvision.models import VGG11_Weights, VGG13_Weights, VGG16_Weights, VGG19_Weights

from src.zoo.fine_tuned.datasets import get_loaders, Cifar
from src.zoo.fine_tuned.fine_tuned_models import TorchVisionModelProvider
from src.zoo.vgg.modeling_cifar import VGGForCIFAR

logger = logging.getLogger()


class TunableVGGProvider(TorchVisionModelProvider):
    VGG11 = models.vgg11, VGGForCIFAR, VGG11_Weights.IMAGENET1K_V1
    VGG13 = models.vgg13, VGGForCIFAR, VGG13_Weights.IMAGENET1K_V1
    VGG16 = models.vgg16, VGGForCIFAR, VGG16_Weights.IMAGENET1K_V1
    VGG19 = models.vgg19, VGGForCIFAR, VGG19_Weights.IMAGENET1K_V1


def _init_optimizer(model):
    return optim.SGD(
        [
            {"params": model.model.features.parameters(), "lr": 1e-4},
            {"params": model.model.classifier.parameters(), "lr": 1e-3},
        ],
        momentum=0.9,
        weight_decay=5e-4,
    )


def vgg_cifar(vgg_model_name: str, cifar: Cifar, load_weights=False):
    try:
        return TunableVGGProvider[vgg_model_name.upper()].model(load_weights, cifar)
    except KeyError as e:
        logger.error(f"Unable to find model {vgg_model_name}")
        raise e


def vgg_for_training(vgg_model_name: str, cifar: Cifar, load_weights=False):
    vgg = vgg_cifar(vgg_model_name, cifar, load_weights)
    return vgg, _init_optimizer(vgg), get_loaders(cifar)
