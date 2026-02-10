from enum import Enum

import timm

from crimson_magick.cifar_zoo.fine_tuned.datasets import Cifar


class ArchType(Enum):
    VGG = "vgg"
    RESNET = "resnet"
    MOBILENET = "mobilenet"


class Arch(Enum):
    VGG11 = "vgg11"
    VGG13 = "vgg13"
    VGG16 = "vgg16"
    VGG19 = "vgg19"
    RESNET18 = "resnet18"
    RESNET34 = "resnet34"
    RESNET50 = "resnet50"
    MOBILENETV1 = "mobilenetv1"
    MOBILENETV2 = "mobilenetv2"


class TunableModelProvider(Enum):

    def model(self, load_weights, cifar: Cifar):
        pass

    def arch_name(self):
        return self.name.lower()

    def model_name(self, cifar: Cifar):
        return f"{self.arch_name()}_{cifar.name.lower()}"


class TorchVisionModelProvider(TunableModelProvider):

    def model(self, load_weights, cifar: Cifar):
        model_constructor, fine_tuned_constructor, weights = self.value
        model_instance = model_constructor(weights=weights if load_weights else None)
        return fine_tuned_constructor(model_instance, self.model_name(cifar), cifar, self.arch_name())


class TIMMModelProvider(TunableModelProvider):

    def model(self, load_weights, cifar: Cifar):
        model_name, fine_tuned_constructor = self.value
        model_instance = timm.create_model(model_name, pretrained=load_weights)
        return fine_tuned_constructor(model_instance, self.model_name(cifar), cifar, self.arch_name())
