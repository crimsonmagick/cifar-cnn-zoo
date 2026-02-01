import logging

from fine_tuned.datasets import CIFAR
from mobilenet import mobilenet_for_training
from resnet import resnet_for_training
from vgg import vgg_for_training

logger = logging.getLogger()


def model_for_training(model_name, dataset_name, load_weights=False):
    cifar_type = dataset_name[5:]
    if cifar_type == "100":
        cifar = CIFAR.CIFAR100
    elif cifar_type == "10":
        cifar = CIFAR.CIFAR10
    else:
        msg = f'Unrecognized dataset "{dataset_name}"'
        logger.error(msg)
        raise RuntimeError(msg)
    str_start = model_name[:3]
    if str_start.upper() == "VGG":
        return vgg_for_training(model_name, cifar, load_weights)
    elif str_start.upper() == "RES":
        return resnet_for_training(model_name, cifar, load_weights)
    elif str_start.upper() == "MOB":
        return mobilenet_for_training(model_name, cifar, load_weights)
    else:
        msg = f'Unrecognized model "{model_name}"'
        logger.error(msg)
        raise RuntimeError(msg)
