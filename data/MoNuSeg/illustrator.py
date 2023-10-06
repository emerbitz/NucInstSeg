from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from skimage.color import label2rgb
from torch import Tensor

from data.MoNuSeg.ground_truth import NucleiInstances
from data.MoNuSeg.utils import cuda_tensor_to_ndarray, threshold


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
    def from_tensor(img: Optional[Tensor] = None, inst: Optional[Tensor] = None,
                    thresh: Union[None, str, float] = None) -> "Picture":
        """
        Creates a Picture from the tensor(s).

        If only the image (img) is given, the Picture is created from this image. If additionally a threshold (thresh)
        is given, the Picture is the thresholded image.

        If only the nuclei instances (inst) are provided, the Picture is a color-coded image of the nuclei instances.
        If an image is provided as well, the Picture is an overlay of image with the color-coded instances.
        """
        if thresh is not None:
            img = Picture.create_mask(img, thresh)
        if inst is not None:
            img = Picture.create_colored_inst(inst, img)  # Inst of shape (#nuclei, H, W)
        elif img is not None:
            img = Picture.tensor_to_ndarray(img)
        else:
            raise ValueError(f"Please provide input tensor for img or inst.")
        return Picture(img)

    @staticmethod
    def create_colored_inst(inst: Tensor, img: Optional[Tensor] = None) -> np.ndarray:
        """
        Creates a color-coded image of the nuclei instances (inst).

        If an image (img) is provided, then the color-coded instances are painted over the image.
        """
        if img is not None:
            img = Picture.tensor_to_ndarray(img)
        labeled_inst = NucleiInstances.from_inst(
            inst).to_labeled_inst()  # Maybe more direct implementation would be better
        return label2rgb(label=labeled_inst, image=img)


    @staticmethod
    def create_mask(img: Tensor, thresh: Union[str, float] = 0.5) -> np.ndarray:
        """
        Creates a mask via thresholding.
        """
        img = Picture.tensor_to_ndarray(img)
        return threshold(img, thresh)

    @staticmethod
    def tensor_to_ndarray(img: Tensor) -> np.ndarray:
        """Converts a tensor into np.ndarray."""
        img = cuda_tensor_to_ndarray(img)

        if Picture._has_channel(img):
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
    def _has_channel(img: np.ndarray) -> bool:
        return img.ndim == 3
