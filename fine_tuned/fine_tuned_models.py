from enum import Enum

import timm

from fine_tuned.datasets import CIFAR

class ArchType(Enum):
    VGG = "vgg"
    RESNET = "resnet"
    MOBILENET = "mobilenet"

class TunableModelProvider(Enum):

    def model(self, load_weights, cifar: CIFAR):
        pass

    def arch_name(self):
        return self.name.lower()

    def model_name(self, cifar: CIFAR):
        return f"{self.arch_name()}_{cifar.name.lower()}"


class TorchVisionModelProvider(TunableModelProvider):

    def model(self, load_weights, cifar: CIFAR):
        model_constructor, fine_tuned_constructor, weights = self.value
        model_instance = model_constructor(weights=weights if load_weights else None)
        return fine_tuned_constructor(model_instance, self.model_name(cifar), cifar, self.arch_name())


class TIMMModelProvider(TunableModelProvider):

    def model(self, load_weights, cifar: CIFAR):
        model_name, fine_tuned_constructor = self.value
        model_instance = timm.create_model(model_name, pretrained=load_weights)
        return fine_tuned_constructor(model_instance, self.model_name(cifar), cifar, self.arch_name())
