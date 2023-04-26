import numpy as np
from torch import Tensor
from typing import Tuple
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.color import label2rgb

from data.MoNuSeg.ground_truth import NucleiInstances

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
    def from_tensor(img: Tensor, inst: Tensor = None) -> "Picture":
        """Creates the Picture from the tensor image (img). If the tensor nuclei instances (inst) is provided as well,
        the Picture is created from the overlay of the image with the nuclei instances."""
        if inst is not None:
            img = Picture.overlay(img, inst)  # inst: shape (#nuclei, H, W)
        else:
            img = Picture.tensor_to_ndarray(img)
        return Picture(img)

    @staticmethod
    def overlay(img: Tensor, inst: Tensor) -> np.ndarray:
        """Overlays the tensor image (img) with the tensor nuclei instances (inst)."""
        img = Picture.tensor_to_ndarray(img)
        labeled_inst = NucleiInstances.from_inst(inst).to_labeled_inst()  # Maybe more direct implementation would be better
        return label2rgb(label=labeled_inst, image=img)

    @staticmethod
    def tensor_to_ndarray(img: Tensor) -> np.array:
        """Converts a tensor into np.ndarray."""
        dim = img.dim()
        if dim == 3:
            channels, *_ = img.size()
            if channels == 3:
                return img.permute((1, 2, 0)).numpy()  # Conversion for RGB image (3, H, W)
            elif channels == 1:
                return img[0].numpy()  # Conversion for grayscale image (1, H, W)
            else:
                raise ValueError(f"Img should have one or three channels. Got instead {channels} channels.")
        elif dim == 2:
            return img.numpy()  # Conversion for channel-less image (H, W)
        else:
            raise ValueError(f"Img should have two or three dimensions. Got instead {img.dim()} dimensions.")

    def show(self) -> None:
        """Displays the Picture."""
        plt.imshow(self.data, cmap="gray")  # Cmap argument is ignored for RGB data per default
        plt.axis("off")
        plt.show()

    def save(self, file_name: str, save_dir: str = "images"):
        """Saves the Picture as png."""
        save_dir = Path(save_dir)
        file = Path(save_dir, file_name + ".png")
        if not save_dir.is_dir():
            save_dir.mkdir()
        plt.imsave(file, arr=self.data, cmap="gray")  # Cmap argument is ignored for RGB data per default
