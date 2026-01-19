import torch
from torch import nn
from torchvision.models import VGG

CLASS_COUNT = 10

class VGGForCIFAR10(nn.Module):

    def __init__(self, vgg: VGG, model_name):
        super().__init__()
        self.model = vgg
        self._model_name = model_name
        out_idx = len(self.model.classifier) - 1
        feature_count = self.model.classifier[out_idx].in_features
        self.model.classifier[out_idx] = nn.Linear(feature_count, CLASS_COUNT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @property
    def model_name(self):
        return self._model_name

