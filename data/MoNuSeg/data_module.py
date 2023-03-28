import random
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split

from data.MoNuSeg.dataset import MoNuSeg
from data.MoNuSeg.dataset_creator import MoNuSegCreator
from data.MoNuSeg.dataset_patcher import MoNuSegPatcher
from data.MoNuSeg.utils import custom_collate
from augmentation.augmentations import RandCrop, RandHorizontalFlip, RandRotate, RandVerticalFlip
from transformation.transformations import Combine, ToTensor, PadZeros


class MoNuSegDataModule(pl.LightningDataModule):
    """
    Performs splitting and batching into train, validation and test for the MoNuSeg dataset
    """

    def __init__(self, seg_masks: bool = True, cont_masks: bool = True, dist_maps: bool = True, labels: bool = False,
                 data_root: str = "datasets"):
        super().__init__()
        self.seg_masks = seg_masks
        self.cont_masks = cont_masks
        self.dist_maps = dist_maps
        self.labels = labels
        # self.instances = instances
        self.root = data_root

        self.batch_size = 4
        self.img_size = (256, 256)
        self.threads = torch.get_num_threads()

        self.train_transforms = Combine([
            # ToTensor(),
            RandHorizontalFlip(p=0.5),
            RandVerticalFlip(p=0.5),
            RandRotate(degrees=360.),
            # RandCrop(size=(256, 256)),
        ])
        self.val_transforms = None
        self.test_transforms = None

        self.train_data: MoNuSeg
        self.val_data: MoNuSeg
        self.test_data: MoNuSeg

    def prepare_data(self) -> None:
        creator = MoNuSegCreator(root=self.root)
        creator.save_ground_truths(
            segmentation_masks=self.seg_masks,
            contour_masks=self.cont_masks,
            distance_maps=self.dist_maps
        )
        patcher = MoNuSegPatcher(
            dataset=MoNuSeg(
                root=self.root,
                segmentation_masks=self.seg_masks,
                contour_masks=self.cont_masks,
                distance_maps=self.dist_maps,
                instances=True,
                transforms=ToTensor(),
                dataset="Train Kaggle"
            )
        )
        patcher.split_and_save(patch_size=self.img_size)

    def setup(self, stage: str = None) -> None:
        if stage == "fit" or stage is None:
            # Splitting of train dataset into train and validation dataset
            training_dataset = MoNuSeg.select_data(dataset="Train")
            random.shuffle(training_dataset)
            train_dataset, val_dataset = training_dataset[:12], training_dataset[12:]

            self.train_data = MoNuSeg(
                root=self.root,
                segmentation_masks=self.seg_masks,
                contour_masks=self.cont_masks,
                distance_maps=self.dist_maps,
                labels=self.labels,
                instances=False,
                dataset=train_dataset,
                transforms=self.train_transforms,
                size="256"
            )
            self.val_data = MoNuSeg(
                root=self.root,
                segmentation_masks=self.seg_masks,
                contour_masks=self.cont_masks,
                distance_maps=self.dist_maps,
                labels=self.labels,
                instances=True,
                dataset=val_dataset,
                transforms=self.val_transforms,
                size="256"
            )

        if stage == "test" or stage is None:
            self.test_data = MoNuSeg(
                root=self.root,
                segmentation_masks=self.seg_masks,
                contour_masks=self.cont_masks,
                distance_maps=self.dist_maps,
                labels=self.labels,
                instances=True,
                dataset="Test",
                transforms=self.test_transforms,
                size="256"
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.threads, collate_fn=custom_collate)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.threads, collate_fn=custom_collate)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.threads, collate_fn=custom_collate)
