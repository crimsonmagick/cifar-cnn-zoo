import timm
from timm.models import EfficientNet
from torch import optim

from vgg.modeling_cifar10 import VGGForCIFAR10

def _init_optimizer(model: EfficientNet):

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

def mobilenet_v1_cifar10(transfer_learn=False):
    model: EfficientNet = timm.create_model("mobilenetv1_100", pretrained=transfer_learn)
    return model, _init_optimizer(model)

