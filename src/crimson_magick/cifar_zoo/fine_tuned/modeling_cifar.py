import torch
from torch import nn

from crimson_magick.cifar_zoo.fine_tuned.datasets import Cifar
from crimson_magick.cifar_zoo.fine_tuned.fine_tuned_models import ArchType


class CifarCNN(nn.Module):

    def __init__(self, original_model: nn.Module, cifar: Cifar, model_name: str, arch_type: ArchType, arch_name: str):
        super().__init__()
        self.model = original_model
        self.dataset = cifar
        self.model_name = model_name
        self.arch_type = arch_type
        self.arch_name = arch_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
