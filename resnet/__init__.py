import logging

from torch import optim
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

from fine_tuned.datasets import get_loaders, CIFAR
from fine_tuned.fine_tuned_models import TorchVisionModelProvider
from resnet.modeling_cifar import ResNetForCIFAR

logger = logging.getLogger()


class TunableResnetProvider(TorchVisionModelProvider):
    RESNET18 = models.resnet18, ResNetForCIFAR, ResNet18_Weights.IMAGENET1K_V1
    RESNET34 = models.resnet34, ResNetForCIFAR, ResNet34_Weights.IMAGENET1K_V1
    RESNET50 = models.resnet50, ResNetForCIFAR, ResNet50_Weights.IMAGENET1K_V1


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


def resnet_cifar(resnet_model_name: str, cifar: CIFAR, load_weights=False):
    try:
        return TunableResnetProvider[resnet_model_name.upper()].model(load_weights, cifar)
    except KeyError as e:
        logger.error(f"Unable to find model {resnet_model_name}")
        raise e


def resnet_for_training(resnet_model_name: str, cifar: CIFAR, load_weights=False):
    res_net = resnet_cifar(resnet_model_name, cifar, load_weights)
    return res_net, _init_optimizer(res_net), get_loaders(cifar)
