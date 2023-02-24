import unittest
from torchvision.transforms import functional as F

from test_augmentation_base import AugmentationTest
from augmentations import RandHorizontalFlip, RandVerticalFlip, RandRotate

class TestRandHorizontalFlip(AugmentationTest, unittest.TestCase):
    def transform_img(self):
        return F.hflip(self.img)

    def augment_imgs(self, p: float):
        return RandHorizontalFlip(p)(self.imgs)

class TestRandVerticalFlip(AugmentationTest, unittest.TestCase):
    def transform_img(self):
        return F.vflip(self.img)

    def augment_imgs(self, p: float):
        return RandVerticalFlip(p)(self.imgs)


if __name__ == '__main__':
    unittest.main()
