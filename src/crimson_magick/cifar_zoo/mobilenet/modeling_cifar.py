from timm.models import EfficientNet
from torch import nn

from src.crimson_magick.cifar_zoo.fine_tuned.datasets import Cifar
from src.crimson_magick.cifar_zoo.fine_tuned.fine_tuned_models import ArchType
from src.crimson_magick.cifar_zoo.fine_tuned.modeling_cifar import CifarCNN


class MobileNetForCIFAR(CifarCNN):

    def __init__(self, mobilenet: EfficientNet, model_name, cifar: Cifar, arch_name: str):
        super().__init__(mobilenet, cifar, model_name, ArchType.MOBILENET, arch_name)
        in_feature_count = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_feature_count, cifar.value)
