import timm
from timm.models import EfficientNet
from torch import optim

from fine_tuned.datasets import CIFAR, get_loaders
from mobilenet.modeling_cifar import MobileNetForCIFAR


def _init_optimizer(model: MobileNetForCIFAR):
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


def mobilenet_v1_cifar100(transfer_learn=False):
    mobilenet: EfficientNet = timm.create_model("mobilenetv1_100", pretrained=transfer_learn)
    model = MobileNetForCIFAR(mobilenet, "mobilenet_v1_cifar100", CIFAR.CIFAR100)
    return model, _init_optimizer(model), get_loaders(CIFAR.CIFAR100)


def mobilenet_v2_cifar100(transfer_learn=False):
    mobilenet: EfficientNet = timm.create_model("mobilenetv2_100", pretrained=transfer_learn)
    model = MobileNetForCIFAR(mobilenet, "mobilenet_v2_cifar100", CIFAR.CIFAR100)
    return model, _init_optimizer(model), get_loaders(CIFAR.CIFAR100)
