import json
import logging
import re

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file

from constants import HUGGINGFACE_ZOO
from fine_tuned.datasets import CIFAR, get_test_loader
from mobilenet import mobilenet_for_training, mobilenet_cifar
from resnet import resnet_for_training, resnet_cifar
from vgg import vgg_for_training, vgg_cifar

logger = logging.getLogger()


def get_cifar(dataset_name):
    cifar_type = dataset_name[5:]
    if cifar_type == "100":
        return CIFAR.CIFAR100
    elif cifar_type == "10":
        return CIFAR.CIFAR10
    else:
        msg = f'Unrecognized dataset "{dataset_name}"'
        logger.error(msg)
        raise RuntimeError(msg)


def model_cifar(model_name, dataset_name, load_weights=False):
    cifar = get_cifar(dataset_name)
    str_start = model_name[:3]
    if str_start.upper() == "VGG":
        return vgg_cifar(model_name, cifar, load_weights)
    elif str_start.upper() == "RES":
        return resnet_cifar(model_name, cifar, load_weights)
    elif str_start.upper() == "MOB":
        return mobilenet_cifar(model_name, cifar, load_weights)
    else:
        msg = f'Unrecognized model "{model_name}"'
        logger.error(msg)
        raise RuntimeError(msg)


def model_for_training(model_name, dataset_name, load_weights=False):
    cifar = get_cifar(dataset_name)
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


def _is_safetensor(path):
    expr = re.compile('.+\.safetensors$')
    return True if expr.match(path) else False


def _model_from_safetensor(model_dir, device):
    config_path = f"{model_dir}/config.json"
    model_path = f"{model_dir}/model.safetensors"

    with open(config_path, "r") as f:
        config = json.load(f)

    arch_name = config['arch_name']
    dataset_name = config['dataset']
    model = model_cifar(arch_name, dataset_name, False)
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model_state = load_file(model_path)
    model.load_state_dict(model_state)
    return model, dataset_name



def model_from_hf_hub(model_name: str, device):
    model_dir = snapshot_download(repo_id=HUGGINGFACE_ZOO, allow_patterns=f"{model_name}")
    return _model_from_safetensor(model_dir, device)


def _model_from_checkpoint(checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
        base_model_name = checkpoint['base_model_name']
        dataset_name = checkpoint['dataset_name']
        model = model_cifar(base_model_name, dataset_name, False)
        model = model.to(device)
        for param in model.parameters():
            param.requires_grad = False
        model.load_state_dict(checkpoint['model_state'])
    except FileNotFoundError as e:
        logger.error(f"Checkpoint {checkpoint_path} was not found, exception={repr(e)} Exiting...")
        raise e
    except Exception as e:
        logger.error(
            f"Encountered exception while loading checkpoint {checkpoint_path}, exception={repr(e)}. Exiting...")
        raise e
    return model, dataset_name


def model_from_checkpoint(checkpoint_path, device):
    model, _ = _model_from_checkpoint(checkpoint_path, device)
    return model


def model_for_testing(checkpoint_path, device):
    model, dataset_name = _model_from_checkpoint(checkpoint_path, device)
    cifar = get_cifar(dataset_name)
    return model, get_test_loader(cifar)

def model_for_testing_safetensor(model_path, device):
    model, dataset_name = _model_from_safetensor(model_path, device)
    cifar = get_cifar(dataset_name)
    return model, get_test_loader(cifar)