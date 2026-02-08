from torch import nn
from torchvision.models import ResNet

from fine_tuned.datasets import CIFAR
from fine_tuned.modeling_cifar import CifarCNN

class ResNetForCIFAR(CifarCNN):

    def __init__(self, resnet: ResNet, model_name, cifar: CIFAR):
        super().__init__(resnet, cifar, model_name)
        in_feature_count = self.model.fc.in_features
        self.model.fc = nn.Linear(in_feature_count, cifar.value)

