import random
from abc import ABC, abstractmethod
from typing import Dict

from torch import Tensor


class Augmentation(ABC):
    """Base class for image augmentations"""

    @abstractmethod
    def __call__(self, gt: Dict[str, Tensor]) -> Dict[str, Tensor]:
        pass


class AugmentAll(Augmentation):
    """Base class for image augmentations to be applied to all ground truths."""

    def __call__(self, gt: Dict[str, Tensor]) -> Dict[str, Tensor]:
        seed = random.random()
        for key, value in gt.items():
            gt[key] = self.transform(value, seed)
        return gt

    @abstractmethod
    def transform(self, img: Tensor, seed: float) -> Tensor:
        pass


class AugmentOne(Augmentation):
    """Base class for image augmentations to be applied to only one ground truth."""

    def __init__(self, name: str = "img"):
        self.name = name

    def __call__(self, gt: Dict[str, Tensor]) -> Dict[str, Tensor]:
        gt[self.name] = self.transform(gt[self.name])
        return gt

    @abstractmethod
    def transform(self, img: Tensor) -> Tensor:
        pass
