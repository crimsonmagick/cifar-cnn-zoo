import torch
from torch import nn
from torchvision.models import VGG, ResNet

CLASS_COUNT = 10

class ResNetForCIFAR10(nn.Module):

    def __init__(self, resnet: ResNet, model_name):
        super().__init__()
        self.model = resnet
        self._model_name = model_name
        in_feature_count = self.model.fc.in_features
        self.model.fc = nn.Linear(in_feature_count, CLASS_COUNT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @property
    def model_name(self):
        return self._model_name

