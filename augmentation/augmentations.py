import random
from typing import Tuple, Union
import torch
import torchvision.transforms.functional as TF

from augmentation.augmentation_base import Augmentation
from transformation.utils import remove_zero_stacks

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
        rotated = TF.rotate(img, angle=angle)
        return remove_zero_stacks(rotated)


class RandCrop(Augmentation):
    """
    Generates a random crop from the image of size (h, w).
    """

    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def transform(self, img: torch.Tensor, seed: float) -> Union[torch.Tensor, None]:
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


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import torchvision.transforms as T
    from data.MoNuSeg.dataset import MoNuSeg
    from data.MoNuSeg.illustrator import Picture
    from transformation.transformations import ToTensor, Combine, Split, Resize, PadZeros
    from data.MoNuSeg.data_module import MoNuSegDataModule

    transforms = Combine([
        # ToTensor(),
        RandVerticalFlip(p=0.5),
        RandHorizontalFlip(p=0.5),
        # RandRotate(degrees=360.),
        # PadZeros(padding=12),
        # Split(size=(256,256))
    ])

    train_data = MoNuSeg(root="../datasets", dataset="Train", instances=True, labels=True, size="256", transforms=transforms)
    pair = train_data[0]
    for element in pair:
        if isinstance(element, torch.Tensor):
            channels, *_ = element.shape
            if channels > 3:
                print(f"Number of instances: {channels}")
                # for img in element:
                #     Picture.from_tensor(img).show()
            else:
                Picture.from_tensor(element).show()
        elif isinstance(element, str):
            print(element)
        else:
            print(type(element))

    # data = MoNuSeg(root="../datasets", split="Test", transforms=transforms)
    # loader = DataLoader(dataset=data, batch_size=8, shuffle=False)
    # imgs, seg_masks, cont_masks, *_ = next(iter(loader))
    # print(imgs[0].shape)
    # pic = Picture.from_tensor(data[0][1])
    # print(pic.size())
    # pic.show()

    # data_module = MoNuSegDataModule(data_root="../datasets")
    # data_module.prepare_data()
    # data_module.setup(stage="fit")
    # train_loader = data_module.train_dataloader()
    # img, *_ = next(iter(train_loader))
    # print(img.size())
    # val_loader = data_module.val_dataloader()
    # img, *_ = next(iter(val_loader))
    # print(img.size())
    # for i in range(10):
    #     for batch in dataloader:
    #         imgs, *_ = batch
    #         element = imgs[0,:,:,:]
    #         pic = Picture.from_tensor(element)
    #         print(pic.size())
    #         pic.show()
