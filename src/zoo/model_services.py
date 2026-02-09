import json
import logging

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from src.zoo.constants import HUGGINGFACE_ZOO
from src.zoo.fine_tuned.datasets import Cifar, get_test_loader
from src.zoo.fine_tuned.fine_tuned_models import Arch
from src.zoo.mobilenet import mobilenet_cifar, mobilenet_for_training
from src.zoo.resnet import resnet_cifar, resnet_for_training
from src.zoo.vgg import vgg_cifar, vgg_for_training

logger = logging.getLogger()

valid_models = [
    "mobilenetv1_cifar100",
    "mobilenetv2_cifar100",
    "resnet18_cifar10",
    "resnet34_cifar10",
    "resnet50_cifar100",
    "vgg11_cifar10",
    "vgg13_cifar10",
    "vgg16_cifar10",
    "vgg19_cifar10"
]


def get_cifar(dataset_name):
    cifar_type = dataset_name[5:]
    if cifar_type == "100":
        return Cifar.CIFAR100
    elif cifar_type == "10":
        return Cifar.CIFAR10
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
    return model


def model_from_hf_hub(model_name: str, device):
    base_dir = snapshot_download(repo_id=HUGGINGFACE_ZOO, allow_patterns=f"{model_name}/*")
    return _model_from_safetensor(f"{base_dir}/{model_name}", device)


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


def load_model(arch: Arch, dataset: Cifar, device):
    model_name = f"{arch.value.lower()}_{dataset.name.lower()}"
    if model_name not in valid_models:
        error_message = f"Invalid model/arch combination: '{model_name}'. Valid models/arch combinations: {valid_models}"
        logger.error(error_message)
        raise RuntimeError(error_message)

    return model_from_hf_hub(model_name, device)
