import numpy as np
import PIL
from typing import Any, List, Union
import torch
import torchvision.transforms as T

from augmentation.augmentation_base import Augmentation
from transformation.transformation_base import Transformation


class Combine:
    """Allows for the combination of Augmentation and Transformation subclasses"""
    def __init__(self, transforms: List[Union[Augmentation, Transformation]]):
        self.transforms = transforms

    def __call__(self, imgs: List[Any]) -> List[Any]:
        for transform in self.transforms:
            imgs = transform(imgs)
        return imgs


class ToTensor(Transformation):
    """Converts a np.ndarray or PIL.Image.Image object to a torch.Tensor object"""
    def transform(self, img: Union[np.ndarray, PIL.Image.Image]) -> torch.Tensor:
        return T.ToTensor()(img)
