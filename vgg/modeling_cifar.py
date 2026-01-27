from torch import nn
from torchvision.models import VGG

from fine_tuned.datasets import CIFAR
from fine_tuned.modeling_finetuned import FineTunedCNN

CLASS_COUNT = 10


class VGGForCIFAR(FineTunedCNN):

    def __init__(self, vgg: VGG, model_name, cifar: CIFAR):
        super().__init__(vgg, model_name)
        head_index = len(self.model.classifier) - 1
        in_feature_count = self.model.classifier[head_index].in_features
        self.model.classifier[head_index] = nn.Linear(in_feature_count, cifar.value)
