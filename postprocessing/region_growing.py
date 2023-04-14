from typing import Tuple
import torch
from torch import Tensor


class RegionGrower:
    """Region Growing implementation"""

    def __init__(self, mask):
        """
        :param mask: Segmentation mask of shape (1, H, W)
        """
        self.device = mask.device
        self.mask = mask.squeeze()  # Squeezes tensor to shape (H, W)
        self.height, self.width = self.mask.shape

    def get_regions(self) -> Tensor:
        """Extracts regions (i.e., nuclei instances) from the segmentation mask."""
        regions = []
        for y in range(self.height):
            for x in range(self.width):
                if self.mask[y, x]:
                    regions.append(self._grow(seed=(y, x)))
        return torch.stack(regions)

    def _grow(self, seed: Tuple[int, int]) -> Tensor:
        """Grows a region from the seed point."""
        region = torch.zeros(size=(self.height, self.width), dtype=torch.bool, device=self.device)
        region[seed] = True
        neighborhood = [seed]
        for pixel in neighborhood:
            y_min = max(0, pixel[0]-1)
            x_min = max(0, pixel[1]-1)
            y_max = min(pixel[0] + 2, self.height)
            x_max = min(pixel[1] + 2, self.width)
            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    if self.mask[y, x]:
                        region[y, x] = True
                        self.mask[y, x] = False
                        neighborhood.append((y, x))
        return region



