import numpy as np
from torch import Tensor
from typing import NoReturn, Tuple
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
        """Converts an image as torch.Tensor into an instance of the Picture class"""
        img = Picture.tensor_to_array(img)
        return Picture(img)

    @staticmethod
    def tensor_to_array(img: Tensor) -> np.array:
        """Converts torch.Tensor into np.ndarray"""

        c_channels = img.size()[0]
        if c_channels > 1:
            return img.permute((1, 2, 0)).numpy()  # Conversion for RGB image
        return img[0].numpy()  # Conversion for grayscale image

    def show(self) -> NoReturn:
        """Displays the image"""
        plt.imshow(self.data, cmap="gray")
        plt.axis("off")
        plt.show()
