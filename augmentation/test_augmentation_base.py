import torch
from abc import ABC, abstractmethod

from data.MoNuSeg.dataset import MoNuSeg

class AugmentationTest(ABC):
    """Base class for unit testing of augmentations"""
    img = MoNuSeg(root="../datasets")[0][0]
    imgs = [img] * 10

    @abstractmethod
    def transform_img(self):
        pass

    @abstractmethod
    def augment_imgs(self, p: float):
        pass

    def test_no_augmentation(self):
        """Tests for the application of no augmentation"""
        augmented_imgs = self.augment_imgs(p=0.)
        for augmented in augmented_imgs:
            self.assertTrue(torch.equal(augmented, self.img))

    def test_augmentation(self):
        """Tests for the application of the augmentation"""
        augment_imgs = self.augment_imgs(p=1.)
        for augmented in augment_imgs:
            self.assertTrue(torch.equal(augmented, self.transform_img()))

    def test_same_augmentation(self):
        """Tests for the application of the same augmentation to all images"""
        augment_imgs = self.augment_imgs(p=0.5)
        if torch.equal(augment_imgs[0], self.img):
            is_transformed = False
        else:
            is_transformed = True
        for augmented in augment_imgs:
            if is_transformed:
                self.assertTrue(torch.equal(augmented, self.transform_img()))
            else:
                self.assertTrue(torch.equal(augmented, self.img))

# class AugmentationTest(ABC):
#     img = MoNuSeg(root="../datasets")[0][0]
#     imgs = [img] * 10
#
#     @staticmethod
#     @abstractmethod
#     def transform(img):
#         pass
#
#     @staticmethod
#     @abstractmethod
#     def augment(imgs, p):
#         pass
#
#     def test_no_augmentation(self):
#         aug_imgs = AugmentationTest.augment(self.imgs, p=0)
#         for aug_img in aug_imgs:
#             self.assertTrue(torch.equal(aug_img, self.img))
#
#     def test_augmentation(self):
#         aug_imgs = AugmentationTest.augment(self.imgs, p=1)
#         for aug_img in aug_imgs:
#             self.assertTrue(torch.equal(aug_img, AugmentationTest.transform(self.img)))
#
#     def test_same_augmentation(self):
#         aug_imgs = AugmentationTest.augment(self.imgs, p=0.5)
#         if torch.equal(aug_imgs[0], self.img):
#             is_transformed = False
#         else:
#             is_transformed = True
#         for aug_img in aug_imgs:
#             if is_transformed:
#                 self.assertTrue(torch.equal(aug_img,  AugmentationTest.transform(self.img)))
#             else:
#                 self.assertTrue(torch.equal(aug_img, self.img))
#
