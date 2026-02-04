import logging

from torch import optim

from fine_tuned.datasets import CIFAR, get_loaders
from fine_tuned.fine_tuned_models import TIMMModelProvider
from mobilenet.modeling_cifar import MobileNetForCIFAR

logger = logging.getLogger()


class TunableMobilenetProvider(TIMMModelProvider):
    MOBILENETV1 = "mobilenetv1_100", MobileNetForCIFAR
    MOBILENETV2 = "mobilenetv2_100", MobileNetForCIFAR


def _init_optimizer(model):
    base_params = [p for n, p in model.model.named_parameters() if not n.startswith("classifier")]
    head_params = model.model.classifier.parameters()
    return optim.SGD(
        [
            {"params": base_params, "lr": 1e-4},
            {"params": head_params, "lr": 1e-3},
        ],
        momentum=0.9,
        weight_decay=5e-4,
    )


def mobilenet_cifar(mobilenet_model_name: str, cifar: CIFAR, load_weights=False):
    try:
        return TunableMobilenetProvider[mobilenet_model_name.upper()].model(load_weights, cifar)
    except KeyError as e:
        logger.error(f"Unable to find model {mobilenet_model_name}")
        raise e


def mobilenet_for_training(mobilenet_model_name: str, cifar: CIFAR, load_weights=False):
    mobilenet = mobilenet_cifar(mobilenet_model_name, cifar, load_weights)
    return mobilenet, _init_optimizer(mobilenet), get_loaders(cifar)
