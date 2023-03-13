import numpy as np
import PIL
from typing import Any, List, NoReturn, Union, Sequence, Tuple
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as f

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


class Resize(Transformation):
    """Resizes the image to a given size"""

    def __init__(self, size: List[int], interpolation: f.InterpolationMode = f.InterpolationMode.BILINEAR) -> NoReturn:
        self.size = size
        self.interpolation = interpolation

    def transform(self, img: torch.Tensor) -> torch.Tensor:
        return f.resize(img=img, size=self.size, interpolation=self.interpolation)


class Split(Transformation):
    """Splits the image into smaller pieces of given size."""

    def __init__(self, size: Union[Tuple[int, int], int]) -> NoReturn:
        if not isinstance(size, (tuple, int)):
            raise TypeError(f"Size should be of type tuple or int. Got instead '{type(size)}'.")
        if isinstance(size, tuple) and len(size) != 2:
            raise ValueError(
                f"Size as tuple should contain a value for height and width each. Got instead {len(size)} values.")
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def transform(self, img: torch.Tensor) -> List[torch.Tensor]:
        _, img_height, img_width = img.size()
        num_splits_y, num_splits_x = -(img_height // -self.size[0]), -(img_width // -self.size[1])  # Ceiling division
        padding = (num_splits_y * self.size[0] - img_height, num_splits_x * self.size[1] - img_width)
        if padding != (0, 0):
            img = self.pad_zeros(img=img, padding=padding)

        output = []
        for y in range(0, img_height, self.size[0]):
            for x in range(0, img_width, self.size[1]):
                output.append(img[:, y: y + self.size[0], x: x + self.size[1]])
        return output

    @staticmethod
    def pad_zeros(img: torch.Tensor, padding: Union[Tuple[int, int], int]):
        """Pads the image on the height and width axis with the given number of zeros."""
        if not isinstance(padding, (tuple, int)):
            raise TypeError(f"Padding should be of type tuple or int. Got instead '{type(padding)}'.")
        if isinstance(padding, tuple) and len(padding) != 2:
            raise ValueError(
                f"Padding as tuple should contain a value for height and width each. Got instead {len(padding)} values.")
        if isinstance(padding, int):
            padding = [0, 0, padding, padding]
            # padding = (padding, padding)
        else:
            padding = [0, 0, padding[0], padding[1]]
            # padding = (padding[0], padding[1])

        return f.pad(img=img, padding=padding, fill=0.)


class PadZeros(Transformation):
    """Pads the image on all sides with the given number of zeros."""
    def __init__(self, padding: Union[Tuple[int, int], int]):
        if not isinstance(padding, (tuple, int)):
            raise TypeError(f"Padding should be of type tuple or int. Got instead '{type(padding)}'.")
        if isinstance(padding, tuple) and len(padding) != 2:
            raise ValueError(
                f"Padding as tuple should contain a value for height and width each. Got instead {len(padding)} values.")
        if isinstance(padding, int):
            self.padding = [padding, padding]
        else:
            self.padding = [padding[0], padding[1]]

    def transform(self, img: torch.Tensor) -> torch.Tensor:
        return f.pad(img=img, padding=self.padding, fill=0.)


class ToTensor(Transformation):
    """Converts a np.ndarray or PIL.Image.Image object to a torch.Tensor object"""

    def transform(self, img: Union[np.ndarray, PIL.Image.Image]) -> torch.Tensor:
        return T.ToTensor()(img)
