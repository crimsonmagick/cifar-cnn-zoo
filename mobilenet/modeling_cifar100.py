import torch
from timm.models import EfficientNet
from torch import nn

CLASS_COUNT = 100


class MobileNetForCIFAR10(nn.Module):

    def __init__(self, mobilenet: EfficientNet, model_name):
        super().__init__()
        self.model = mobilenet
        self._model_name = model_name
        in_feature_count = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_feature_count, CLASS_COUNT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @property
    def model_name(self):
        return self._model_name
