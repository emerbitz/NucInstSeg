from pathlib import Path

import numpy as np
from tqdm import tqdm

from data.MoNuSeg.conversion import NucleiInstances
from data.MoNuSeg.utils import comp_dirs


class MoNuSegCreator:
    """
    Prepares the MoNuSeg dataset prior to usage
    """

    def __init__(self, root: str = "datasets", ):
        self.base_dir = Path(root, "MoNuSeg 2018")
        # self.img_dir = Path(self.base_dir, "Tissue Images")
        self.label_dir = Path(self.base_dir, "Annotations")

    def save_ground_truths(self, segmentation_masks: bool = True, contour_masks: bool = True,
                           distance_maps: bool = True) -> None:
        """
        Generates and saves the ground truths of the specified types if the ground truths do not exist
        """

        print("##### Generating ground truths #####")
        if segmentation_masks:
            self.save_truth_type("Segmentation masks")
        if contour_masks:
            self.save_truth_type("Contour masks")
        if distance_maps:
            self.save_truth_type("Distance maps")

    @staticmethod
    def make_seg_mask(label: Path) -> np.ndarray:
        """
        Generates the segmentation mask for the specified label
        """
        instances = NucleiInstances.from_MoNuSeg(label)
        return instances.to_seg_mask()

    @staticmethod
    def make_cont_mask(label: Path) -> np.ndarray:
        """
        Generates the contour mask for the specified label
        """
        instances = NucleiInstances.from_MoNuSeg(label)
        return instances.to_cont_mask()

    @staticmethod
    def make_dist_map(label: Path) -> np.ndarray:
        """
        Generates the distance map for the specified label
        """
        instances = NucleiInstances.from_MoNuSeg(label)
        return instances.to_dist_map()

    def save_truth_type(self, truth_type: str) -> None:
        """
        Generates and saves the ground truths of the specified type if the ground truths do not exist.

        Parameters:
          truth_type: "Segmentation masks", "Contour masks", "Distance maps" or "HV distance maps"
        """
        save_dir = Path(self.base_dir, truth_type)
        save_dir.mkdir(exist_ok=True)
        labels = comp_dirs(self.label_dir, save_dir, file_suffix=".xml")
        print(f'Generating: {truth_type}')
        pbar = tqdm(labels)
        for label in pbar:
            pbar.set_description(f"Processing {label.stem}")
            instances = NucleiInstances.from_MoNuSeg(label)
            if truth_type == "Segmentation masks":
                truth = instances.to_seg_mask()
            elif truth_type == "Contour masks":
                truth = instances.to_cont_mask()
            elif truth_type == "Distance maps":
                truth = instances.to_dist_map()
            else:
                raise ValueError(
                    f"Truth type must be 'Segmentation masks', 'Contour masks' or 'Distance maps'. "
                    f"Got instead {truth_type}.")
            save_path = Path(save_dir, label.stem + ".npy")
            np.save(str(save_path), truth)
