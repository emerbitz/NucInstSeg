from pathlib import Path
from typing import List, NoReturn, Tuple, Union

import torch
from tqdm import tqdm

from data.MoNuSeg.dataset import MoNuSeg
from transformation.transformations import Split


class MoNuSegPatcher:
    """Creates image patches from the MoNuSeg dataset."""

    def __init__(self, dataset: MoNuSeg):
        self.data = dataset
        self.data.labels = True  # Required to save patches

        self.dirs = [self.data.img_dir]
        if self.data.segmentation_mask:
            self.dirs.append(self.data.seg_mask_dir)
        if self.data.contour_mask:
            self.dirs.append(self.data.cont_mask_dir)
        if self.data.distance_map:
            self.dirs.append(self.data.dist_map_dir)
        if self.data.instances:
            self.dirs.append(self.data.inst_dir)

    def split_and_save(self, patch_size: Union[Tuple[int, int], int]):
        """Splits all images of the dataset into patches of given size and saves the patches."""
        print("##### Generating image patches #####")
        pbar = tqdm(self.data)
        for *imgs, label in pbar:
            pbar.set_description(f"Processing {label}")
            for img, save_dir in zip(imgs, self.dirs):
                if isinstance(img, torch.Tensor):
                    self.save_splits(img, patch_size=patch_size, save_dir=save_dir, label=label)
                else:
                    raise TypeError(f"Dataset should yield items of type torch.Tensor or str. Got instead {type(img)}")

    @staticmethod
    def save_splits(img: torch.Tensor, patch_size: Union[Tuple[int, int], int], save_dir: Path, label: str) -> NoReturn:
        """Splits the image into patches of given size and saves the patches."""
        if isinstance(patch_size, tuple):
            size = max(patch_size)
        elif isinstance(patch_size, int):
            size = patch_size
        else:
            raise TypeError(f"Split_size should be tuple or int. Got instead {type(patch_size)}.")

        patches = MoNuSegPatcher.split_image(img=img, patch_size=patch_size)
        for num, patch in enumerate(patches):
            save_path = Path(save_dir, label + "_" + str(size) + "_" + str(num) + ".pt")
            if not save_path.is_file():  # Image patch is saved if it does not exist
                MoNuSegPatcher.save_tensor(patch, save_path=save_path)

    @staticmethod
    def split_image(img: torch.Tensor, patch_size: Union[Tuple[int, int], int]) -> List[torch.Tensor]:
        """Splits the image into patches of given size."""
        return Split(size=patch_size).transform(img)

    @staticmethod
    def save_tensor(obj: torch.Tensor, save_path: Path) -> NoReturn:
        """Saves the torch.Tensor object under the specified path."""
        torch.save(obj=obj, f=save_path)