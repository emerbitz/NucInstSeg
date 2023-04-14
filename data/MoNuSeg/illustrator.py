import numpy as np
from torch import Tensor
from typing import Tuple
from pathlib import Path
import matplotlib.pyplot as plt


class Picture:
    """Bundles type conversion and depiction of images"""

    def __init__(self, img: np.array):
        if not isinstance(img, np.ndarray):
            raise TypeError(f"img must be np.ndarray. Got instead '{type(img)}'.")
        self.data = img

    def size(self) -> Tuple[int]:
        """Gives the shape of the image"""
        return self.data.shape

    @staticmethod
    def from_tensor(img: Tensor) -> "Picture":
        """Converts an image as tensor into an instance of the Picture class"""
        img = Picture.tensor_to_ndarray(img)
        return Picture(img)

    @staticmethod
    def tensor_to_ndarray(img: Tensor) -> np.array:
        """Converts tensor into np.ndarray"""
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
        """Displays the image"""
        plt.imshow(self.data, cmap="gray")  # Cmap argument is ignored for RGB data per default
        plt.axis("off")
        plt.show()

    def save(self, file_name: str, save_dir: str = "images"):
        """Saves the image as png"""
        save_dir = Path(save_dir)
        file = Path(save_dir, file_name + ".png")
        if not save_dir.is_dir():
            save_dir.mkdir()
        plt.imsave(file, arr=self.data, cmap="gray")  # Cmap argument is ignored for RGB data per default
