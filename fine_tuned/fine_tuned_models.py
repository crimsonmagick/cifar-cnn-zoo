from enum import Enum

import timm

from fine_tuned.datasets import CIFAR


class TunableModelProvider(Enum):

    def model(self, load_weights, cifar: CIFAR):
        pass

    def _display_name(self, cifar: CIFAR):
        return f"{self.name.lower()}_{cifar.name.lower()}"


class TorchVisionModelProvider(TunableModelProvider):

    def model(self, load_weights, cifar: CIFAR):
        model_constructor, fine_tuned_constructor, weights = self.value
        model_instance = model_constructor(weights=weights if load_weights else None)
        return fine_tuned_constructor(model_instance, self._display_name(cifar), cifar)


class TIMMModelProvider(TunableModelProvider):

    def model(self, load_weights, cifar: CIFAR):
        model_name, fine_tuned_constructor = self.value
        model_instance = timm.create_model(model_name, pretrained=load_weights)
        return fine_tuned_constructor(model_instance, self._display_name(cifar), cifar)
