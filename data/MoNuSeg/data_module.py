from typing import Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from augmentation.augmentations import RandHorizontalFlip, RandRotate, RandVerticalFlip, ColorJitter, GaussianBlur
from data.MoNuSeg.dataset import MoNuSeg
from data.MoNuSeg.utils import custom_collate
from transformation.transformations import Combine


class MoNuSegDataModule(pl.LightningDataModule):
    """
    Performs splitting and batching into train, validation and test for the MoNuSeg dataset.
    """

    @classmethod
    def default_mode(cls, mode: str, auxiliary_task: bool = True, **kwargs) -> "MoNuSegDataModule":
        if mode == "noname":
            return cls(seg_masks=True, cont_masks=True, dist_maps=False, hv_maps=False, **kwargs)
        elif mode in ["baseline", "yang"]:
            return cls(seg_masks=True, cont_masks=auxiliary_task, dist_maps=False, hv_maps=False, **kwargs)
        elif mode == "naylor":
            return cls(seg_masks=False, cont_masks=auxiliary_task, dist_maps=True, hv_maps=auxiliary_task, **kwargs)
        elif mode in ["graham", "exprmtl"]:
            return cls(seg_masks=True, cont_masks=False, dist_maps=False, hv_maps=True, **kwargs)

    def __init__(self, seg_masks: bool = True, cont_masks: bool = True, dist_maps: bool = True, hv_maps: bool = True,
                 labels: bool = False, data_root: str = "datasets", batch_size: int = 8,
                 img_size: Tuple[int, int] = (256, 256),
                 train_transforms=Combine([RandHorizontalFlip(p=0.5),
                                           RandVerticalFlip(p=0.5),
                                           RandRotate(degrees=90.),
                                           GaussianBlur(),
                                           ColorJitter(saturation=0.5),
                                           ]),

                 val_transforms=None,
                 test_transforms=None,
                 test_data_is_val_data: bool = False
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.seg_masks = seg_masks
        self.cont_masks = cont_masks
        self.dist_maps = dist_maps
        self.hv_maps = hv_maps
        self.labels = labels
        self.root = data_root

        self.batch_size = batch_size
        self.img_size = img_size
        self.threads = torch.get_num_threads()

        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.test_data_is_val_data = test_data_is_val_data

        self.train_data: MoNuSeg
        self.val_data: MoNuSeg
        self.test_data: MoNuSeg

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = None) -> None:
        # Splitting of train dataset into train and validation dataset
        # a) Training data according to original publication by Kumar et al. 2017
        training_dataset = MoNuSeg.select_data(dataset="Train")
        # Different organ (i.e., prostate) validation set:
        # train_dataset, val_dataset = training_dataset[:12], training_dataset[12:]
        # Same organ validation set:
        val_dataset = [training_dataset[0], training_dataset[4], training_dataset[9], training_dataset[14]]
        train_dataset = [data for data in training_dataset if data not in val_dataset]
        # b) Training data according to the MoNuSeg 2018 Kaggle challenge
        # train_dataset = MoNuSeg.select_data(dataset="Train")
        # val_dataset = MoNuSeg.select_data(dataset="Test")
        # c) Illegitimate training and validation splits:
        # train_dataset = MoNuSeg.select_data(dataset="Kaggle Train")
        # val_dataset = MoNuSeg.select_data(dataset="Kaggle Test")

        self.train_data = MoNuSeg(
            root=self.root,
            segmentation_masks=self.seg_masks,
            contour_masks=self.cont_masks,
            distance_maps=self.dist_maps,
            hv_distance_maps=self.hv_maps,
            labels=self.labels,
            instances=True,
            dataset=train_dataset,
            transforms=self.train_transforms,
            size="256"
        )
        self.val_data = MoNuSeg(
            root=self.root,
            segmentation_masks=self.seg_masks,
            contour_masks=self.cont_masks,
            distance_maps=self.dist_maps,
            hv_distance_maps=self.hv_maps,
            labels=self.labels,
            instances=True,
            dataset=val_dataset,
            transforms=self.val_transforms,
            size="256"
        )
        self.test_data = MoNuSeg(
            root=self.root,
            segmentation_masks=self.seg_masks,
            contour_masks=self.cont_masks,
            distance_maps=self.dist_maps,
            hv_distance_maps=self.hv_maps,
            labels=self.labels,
            instances=True,
            dataset="Test",
            transforms=self.test_transforms,
            size="256"
        )

        # if stage == "fit" or stage is None:
        # if stage == "test" or stage is None:

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.threads,
                          collate_fn=custom_collate)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.threads,
                          collate_fn=custom_collate)

    def test_dataloader(self) -> DataLoader:
        data = self.val_data if self.test_data_is_val_data else self.test_data
        return DataLoader(data, batch_size=self.batch_size, shuffle=False, num_workers=self.threads,
                          collate_fn=custom_collate)
