import torch


class NucleiSplitter:
    """Splits clustered nuclei through removal of the nuclei contours."""
    def __init__(self, seg: torch.Tensor, cont: torch.Tensor):
        self.seg_mask = seg > 0.5
        self.cont_mask = cont > 0.5

    def split(self) -> torch.Tensor:
        """Removes the contours from the segmented nuclei"""
        return torch.logical_and(self.seg_mask, torch.logical_not(self.cont_mask))
