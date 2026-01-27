import torch
from timm.models import EfficientNet
from torch import nn

from fine_tuned.datasets import CIFAR
from fine_tuned.modeling_finetuned import FineTunedCNN


class MobileNetForCIFAR(FineTunedCNN):

    def __init__(self, mobilenet: EfficientNet, model_name, cifar: CIFAR):
        super().__init__(mobilenet, model_name)
        in_feature_count = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_feature_count, cifar.value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @property
    def model_name(self):
        return self._model_name
