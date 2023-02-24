import random
from typing import Tuple
import torch
import torchvision.transforms.functional as TF

from augmentation.augmentation_base import Augmentation


class RandHorizontalFlip(Augmentation):
    def __init__(self, p: float = 0.5):
        self.p = p

    def transform(self, img: torch.Tensor, seed: float) -> torch.Tensor:
        random.seed(seed)
        if random.random() < self.p:
            return TF.hflip(img)
        return img


class RandVerticalFlip(Augmentation):
    def __init__(self, p: float = 0.5):
        self.p = p

    def transform(self, img: torch.Tensor, seed: float) -> torch.Tensor:
        random.seed(seed)
        if random.random() < self.p:
            return TF.vflip(img)
        return img


class RandRotate(Augmentation):
    """
    Rotates the image by an angle.
    """

    def __init__(self, degrees: float = 360.):
        self.deg = degrees

    def transform(self, img: torch.Tensor, seed: float) -> torch.Tensor:
        random.seed(seed)
        angle = random.random() * self.deg
        return TF.rotate(img, angle=angle)


class RandCrop(Augmentation):
    """
    Generates a random crop from the image of size (h, w).
    """

    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def transform(self, img: torch.Tensor, seed: float) -> torch.Tensor:
        _, h, w = img.size()
        h_max = h - self.size[0]
        w_max = w - self.size[1]
        random.seed(seed)
        h_lower = random.randint(0, h_max)
        h_upper = h_lower + self.size[0]
        w_lower = random.randint(0, w_max)
        w_upper = w_lower + self.size[1]
        random.randint(0, w_max)
        return img[:, h_lower:h_upper, w_lower:w_upper]


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import torchvision.transforms as T
    from data.MoNuSeg.dataset import MoNuSeg
    from data.MoNuSeg.illustrator import Picture
    from transformation.transformation import ToTensor, Combine

    transforms = Combine([
        ToTensor(),
        RandHorizontalFlip(p=0.5),
        RandCrop(size=(10, 10))
    ])

    train_data = MoNuSeg(root="../datasets", split="Train", transforms=transforms)
    for pair in train_data:
        print(type(pair[0]))
        pic = Picture.from_tensor(pair[0])
        print(pic.size())
        pic.show()
