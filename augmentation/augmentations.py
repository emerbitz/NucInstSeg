import random
from typing import Tuple, Union

import torchvision.transforms as t
import torchvision.transforms.functional as TF
from torch import Tensor

from augmentation.augmentation_base import AugmentAll, AugmentOne
from transformation.utils import remove_zero_stacks


class RandHorizontalFlip(AugmentAll):
    def __init__(self, p: float = 0.5):
        self.p = p

    def transform(self, img: Tensor, seed: float) -> Tensor:
        random.seed(seed)
        if random.random() < self.p:
            return TF.hflip(img)
        return img


class RandVerticalFlip(AugmentAll):
    def __init__(self, p: float = 0.5):
        self.p = p

    def transform(self, img: Tensor, seed: float) -> Tensor:
        random.seed(seed)
        if random.random() < self.p:
            return TF.vflip(img)
        return img


class RandRotate(AugmentAll):
    """
    Rotates the image by an angle.
    """

    def __init__(self, degrees: float = 360.):
        self.deg = degrees

    def transform(self, img: Tensor, seed: float) -> Tensor:
        random.seed(seed)
        angle = random.random() * self.deg
        rotated = TF.rotate(img, angle=angle)
        return remove_zero_stacks(rotated)


class RandCrop(AugmentAll):
    """
    Generates a random crop from the image of size (h, w).
    """

    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def transform(self, img: Tensor, seed: float) -> Union[Tensor, None]:
        _, h, w = img.size()
        h_max = h - self.size[0]
        w_max = w - self.size[1]
        random.seed(seed)
        h_lower = random.randint(0, h_max)
        h_upper = h_lower + self.size[0]
        w_lower = random.randint(0, w_max)
        w_upper = w_lower + self.size[1]
        random.randint(0, w_max)
        crop = img[:, h_lower:h_upper, w_lower:w_upper]
        return remove_zero_stacks(crop)


class GaussianBlur(AugmentOne):
    """
    Blurs the image with a gaussian filter.
    """

    def __init__(self, name: str = "img", kernel_size: int = 5, sigma: Tuple[float] = (0.1, 2.0)):
        super().__init__(name)
        self.kernel_size = kernel_size
        self.sigma = sigma

    def transform(self, img: Tensor) -> Tensor:
        return t.GaussianBlur(self.kernel_size, self.sigma)(img)


class ColorJitter(AugmentOne):
    """
    Perturbs the color of the image.
    """

    def __init__(self, name: str = "img", brightness: float = 0., contrast: float = 0., saturation: float = 0.,
                 hue: float = 0.):
        super().__init__(name)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def transform(self, img: Tensor) -> Tensor:
        return t.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)(img)
