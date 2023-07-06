from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.color import label2rgb
from torch import Tensor

from data.MoNuSeg.ground_truth import NucleiInstances
from data.MoNuSeg.utils import cuda_tensor_to_ndarray, prob_to_mask


class Picture:
    """Bundles type conversion and depiction of images."""

    def __init__(self, img: np.array):
        if not isinstance(img, np.ndarray):
            raise TypeError(f"img must be np.ndarray. Got instead '{type(img)}'.")
        self.data = img

    def size(self) -> Tuple[int]:
        """Gives the shape of the Picture."""
        return self.data.shape

    @staticmethod
    def from_tensor(img: Tensor, inst: Optional[Tensor] = None, thresh: Union[None, str, float] = None) -> "Picture":
        """
        Creates the Picture from the tensor image (img).

        If the tensor nuclei instances (inst) is provided in addition to the image, the Picture is created from the
        overlay of the image  with the nuclei instances.
        If a threshold (thresh) is provided in addition to the image, the image is thresholded first.
        """
        if inst is not None:
            img = Picture.create_overlay(img, inst)  # inst: shape (#nuclei, H, W)
        elif thresh is not None:
            img = Picture.create_mask(img, thresh)
        else:
            img = Picture.tensor_to_ndarray(img)
        return Picture(img)

    @staticmethod
    def create_overlay(img: Tensor, inst: Tensor) -> np.ndarray:
        """Overlays the tensor image (img) with the tensor nuclei instances (inst)."""
        img = Picture.tensor_to_ndarray(img)
        labeled_inst = NucleiInstances.from_inst(
            inst).to_labeled_inst()  # Maybe more direct implementation would be better
        return label2rgb(label=labeled_inst, image=img)

    @staticmethod
    def create_mask(img: Tensor, thresh: Union[str, float] = 0.5) -> np.ndarray:
        """
        Creates a mask via thresholding.
        """
        if torch.is_floating_point(img):
            img = torch.sigmoid(img)
        img = Picture.tensor_to_ndarray(img)
        return prob_to_mask(img, thresh)

    @staticmethod
    def tensor_to_ndarray(img: Tensor) -> np.ndarray:
        """Converts a tensor into np.ndarray."""
        img = cuda_tensor_to_ndarray(img)

        if Picture._is_multi_channel(img):
            img = img.transpose((1, 2, 0))

        return img

    def show(self) -> None:
        """Displays the Picture."""
        cmap = self._select_cmap()
        plt.imshow(self.data, cmap=cmap)  # Cmap argument is ignored for RGB data per default
        plt.axis("off")
        plt.show()

    def save(self, file_name: str, save_dir: str = "images"):
        """Saves the Picture as png."""
        save_dir = Path(save_dir)
        file = Path(save_dir, file_name + ".png")
        if not save_dir.is_dir():
            save_dir.mkdir()
        cmap = self._select_cmap()
        plt.imsave(file, arr=self.data, cmap=cmap)  # Cmap argument is ignored for RGB data per default

    def _select_cmap(self) -> str:
        """Selects a color map suitable for the Picture."""
        if self.data.min() < 0:
            cmap = "bwr"
        else:
            cmap = "gray"
        return cmap

    @staticmethod
    def _is_multi_channel(img: np.ndarray) -> bool:
        return img.ndim == 3
