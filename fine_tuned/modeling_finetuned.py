import torch
from torch import nn


class FineTunedCNN(nn.Module):

    def __init__(self, original_model: nn.Module, model_name: str):
        super().__init__()
        self.model = original_model
        self._model_name = model_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @property
    def model_name(self):
        return self._model_name
