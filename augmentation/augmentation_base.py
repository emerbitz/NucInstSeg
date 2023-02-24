from abc import ABC, abstractmethod
from typing import List
import random
import torch


class Augmentation(ABC):
    """Base class for image augmentations"""

    def __call__(self, imgs: List[torch.Tensor]) -> List[torch.Tensor]:
        seed = random.random()
        return [self.transform(img, seed) for img in imgs]

    @abstractmethod
    def transform(self, img: torch.Tensor, seed: float) -> torch.Tensor:
        pass
