from torch import nn
from torchvision.models import VGG

from crimson_magick.cifar_zoo.fine_tuned.datasets import Cifar
from crimson_magick.cifar_zoo.fine_tuned.fine_tuned_models import ArchType
from crimson_magick.cifar_zoo.fine_tuned.modeling_cifar import CifarCNN


class VGGForCIFAR(CifarCNN):

    def __init__(self, vgg: VGG, model_name, cifar: Cifar, arch_name: str):
        super().__init__(vgg, cifar, model_name, ArchType.VGG, arch_name)
        head_index = len(self.model.classifier) - 1
        in_feature_count = self.model.classifier[head_index].in_features
        self.model.classifier[head_index] = nn.Linear(in_feature_count, cifar.value)
