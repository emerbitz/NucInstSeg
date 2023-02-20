from typing import NoReturn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from data.MoNuSeg.dataset import MoNuSeg
from data.MoNuSeg.dataset_creator import MoNuSegCreator


class MoNuSegDataModule(pl.LightningDataModule):
    """
    Performs splitting and batching into train, validation and test for the MoNuSeg dataset
    """

    def __init__(self):
        super().__init__()
        self.seg_masks = True
        self.cont_masks = True
        self.dist_maps = True
        self.batch_size = 4

    def prepare_data(self) -> NoReturn:
        creator = MoNuSegCreator(root="datasets")
        creator.save_ground_truths(
            segmentation_masks=self.seg_masks,
            contour_masks=self.cont_masks,
            distance_maps=self.dist_maps
        )

    def setup(self, stage: str = None) -> NoReturn:
        if stage == "fit" or stage is None:
            data = MoNuSeg(
                root="datasets",
                segmentation_mask=self.seg_masks,
                contour_mask=self.cont_masks,
                distance_map=self.dist_maps,
                split="Train"
            )
            self.train_data, self.val_data = random_split(data, lengths=[12, 4])

        if stage == "test" or stage is None:
            self.test_data = MoNuSeg(
                root="datasets",
                segmentation_mask=self.seg_masks,
                contour_mask=self.cont_masks,
                distance_map=self.dist_maps,
                split="Test"
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
