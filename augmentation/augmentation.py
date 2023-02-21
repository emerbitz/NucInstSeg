import random
from typing import List, Tuple
import torch
import torchvision.transforms.functional as TF

from augmentation_base import Augmentation


class Combine:
    def __init__(self, augmentations: List[Augmentation]):
        self.augmentations = augmentations

    def __call__(self, imgs: List[torch.Tensor]) -> List[torch.Tensor]:
        for augmentation in self.augmentations:
            imgs = augmentation(imgs)
        return imgs


class RandHorizontalFlip(Augmentation):
    def __init__(self, p: float=0.5):
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
        return img[:,h_lower:h_upper, w_lower:w_upper]


if __name__ == "__main__":
    from data.MoNuSeg.dataset import MoNuSeg
    from data.MoNuSeg.illustrator import Picture

    # img = MoNuSeg(root="../datasets")[0][0]
    # flipped = F.hflip(img)
    # rotated = F.rotate(img, angle=360)
    #
    # Picture.from_tensor(img).show()
    # Picture.from_tensor(flipped).show()
    # Picture.from_tensor(rotated).show()

    Picture.from_tensor(MoNuSeg(root="../datasets")[0][0]).show()
    for img in RandCrop(size=(255, 255))(MoNuSeg(root="../datasets")[0]):
        pic = Picture.from_tensor(img)
        print(pic.size())
        pic.show()

    # augments = Combine([
    #     RandVerticalFlip(p=1),
    #     RandHorizontalFlip(p=1),
    #     RandRotate(),
    #     RandCrop(size=(700, 700))
    # ])
    # for img in augments(MoNuSeg(root="../datasets")[0]):
    #     pic = Picture.from_tensor(img)
    #     pic.show()

    # from torch.utils.data import DataLoader
    # import torchvision.transforms as T
    # train_dataset = MoNuSeg(root="../datasets", split="Train")
    # train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    # batch = next(iter(train_dataloader))
    # imgs, seg_masks, cont_masks, dist_maps = batch
    # # # imgs, [seg_masks, cont_masks, dist_maps] = batch
    # # flipped = T.RandomCrop(size=255)(batch)
    # print(len(batch))
    #
