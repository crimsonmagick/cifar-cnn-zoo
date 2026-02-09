from torch import nn
from torchvision.models import ResNet

from src.zoo.fine_tuned.datasets import Cifar
from src.zoo.fine_tuned.fine_tuned_models import ArchType
from src.zoo.fine_tuned.modeling_cifar import CifarCNN


class ResNetForCIFAR(CifarCNN):

    def __init__(self, resnet: ResNet, model_name, cifar: Cifar, arch_name: str):
        super().__init__(resnet, cifar, model_name, ArchType.RESNET, arch_name)
        in_feature_count = self.model.fc.in_features
        self.model.fc = nn.Linear(in_feature_count, cifar.value)
