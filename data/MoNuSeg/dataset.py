from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from data.MoNuSeg.conversion import NucleiInstances


class MoNuSeg(Dataset):
    # Train and test data according to the original publication:
    train = ["TCGA-A7-A13E-01Z-00-DX1",  # Breast
             "TCGA-A7-A13F-01Z-00-DX1",  # Breast
             "TCGA-AR-A1AK-01Z-00-DX1",  # Breast
             "TCGA-AR-A1AS-01Z-00-DX1",  # Breast
             "TCGA-B0-5711-01Z-00-DX1",  # Kidney
             "TCGA-HE-7128-01Z-00-DX1",  # Kidney
             "TCGA-HE-7129-01Z-00-DX1",  # Kidney
             "TCGA-HE-7130-01Z-00-DX1",  # Kidney
             "TCGA-18-5592-01Z-00-DX1",  # Liver
             "TCGA-38-6178-01Z-00-DX1",  # Liver
             "TCGA-49-4488-01Z-00-DX1",  # Liver
             "TCGA-50-5931-01Z-00-DX1",  # Liver
             "TCGA-G9-6336-01Z-00-DX1",  # Prostate
             "TCGA-G9-6348-01Z-00-DX1",  # Prostate
             "TCGA-G9-6356-01Z-00-DX1",  # Prostate; Target for color normalization
             "TCGA-G9-6363-01Z-00-DX1"]  # Prostate

    test = ["TCGA-E2-A1B5-01Z-00-DX1",  # Breast
            "TCGA-E2-A14V-01Z-00-DX1",  # Breast
            "TCGA-B0-5698-01Z-00-DX1",  # Kidney
            "TCGA-B0-5710-01Z-00-DX1",  # Kidney
            "TCGA-21-5784-01Z-00-DX1",  # Liver
            "TCGA-21-5786-01Z-00-DX1",  # Liver
            "TCGA-CH-5767-01Z-00-DX1",  # Prostate
            "TCGA-G9-6362-01Z-00-DX1",  # Prostate
            "TCGA-DK-A2I6-01A-01-TS1",  # Bladder
            "TCGA-G2-A2EK-01A-02-TSB",  # Bladder
            "TCGA-AY-A8YK-01A-01-TS1",  # Colon
            "TCGA-NH-A8F7-01A-01-TS1",  # Colon
            "TCGA-KB-A93J-01A-01-TS1",  # Stomach
            "TCGA-RD-A8N9-01A-01-TS1"]  # Stomach

    # Not part of the original MoNuSeg dataset:
    surplus = ["TCGA-UZ-A9PN-01Z-00-DX1",
               "TCGA-F9-A8NY-01Z-00-DX1",
               "TCGA-MH-A561-01Z-00-DX1",
               "TCGA-XS-A8TJ-01Z-00-DX1",
               "TCGA-UZ-A9PJ-01Z-00-DX1",
               "TCGA-FG-A87N-01Z-00-DX1",
               "TCGA-BC-A217-01Z-00-DX1"]

    # Not part of the original MoNuSeg dataset, test data of the MoNuSeg 2018 Kaggle challenge
    test_kaggle = ["TCGA-2Z-A9J9-01A-01-TS1",
                   "TCGA-44-2665-01B-06-BS6",
                   "TCGA-69-7764-01A-01-TS1",
                   "TCGA-A6-6782-01A-01-BS1",
                   "TCGA-AC-A2FO-01A-01-TS1",
                   "TCGA-AO-A0J2-01A-01-BSA",
                   "TCGA-CU-A0YN-01A-02-BSB",
                   "TCGA-EJ-A46H-01A-03-TSC",
                   "TCGA-FG-A4MU-01B-01-TS1",
                   "TCGA-GL-6846-01A-01-BS1",
                   "TCGA-HC-7209-01A-01-TS1",
                   "TCGA-HT-8564-01Z-00-DX1",
                   "TCGA-IZ-8196-01A-01-BS1",
                   "TCGA-ZF-A9R5-01A-01-TS1"]

    def __init__(self, root: str = "datasets", segmentation_masks: bool = True, contour_masks: bool = True,
                 distance_maps: bool = False, hv_distance_maps: bool = False, instances: bool = True,
                 labels: bool = False, transforms=None, dataset: Union[List[str], str] = "Whole",
                 size: str = "Original"):
        self.segmentation_mask = segmentation_masks
        self.contour_mask = contour_masks
        self.distance_map = distance_maps
        self.hv_distance_map = hv_distance_maps
        self.instances = instances
        self.labels = labels
        self.transforms = transforms

        base_dir = Path(root, "MoNuSeg 2018")
        self.img_dir = Path(base_dir, "Tissue Images")
        self.inst_dir = Path(base_dir, "Annotations")
        self.seg_mask_dir = Path(base_dir, "Segmentation masks")
        self.cont_mask_dir = Path(base_dir, "Contour masks")
        self.dist_map_dir = Path(base_dir, "Distance maps")
        self.hv_map_dir = Path(base_dir, "HV distance maps")

        if isinstance(dataset, str):
            data = self.select_data(dataset)
        elif isinstance(dataset, list):
            data = dataset
        else:
            raise TypeError(f"Dataset should be of type list or str. Got instead {type(dataset)}.")

        if size == "Original":
            self.data = data
        elif size == "256":
            self.data = []
            for label in data:
                for value in range(0, 16):
                    self.data.append(label + "_256_" + str(value))
        else:
            raise ValueError(f"Size should be 'Original' or '256'. Got instead {size}")
        self.size = size

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves the image and the associated ground truth(s)
        """
        file = self.data[idx]
        output = {}

        if self.size == "Original":
            output = self._get_item_original(file=file, output=output)
        elif self.size == "256":
            output = self._get_item_256(file=file, output=output)

        if self.transforms is not None:
            output = self.transforms(output)

        if self.hv_distance_map:
            inst = output["inst"]
            if isinstance(inst, Tensor):
                hv_map = NucleiInstances.from_inst(inst).to_hv_map(order="CHW")  # Shape (C, H, W)
                hv_map = torch.from_numpy(hv_map)
            elif isinstance(inst, List):
                hv_map = NucleiInstances(inst).to_hv_map(order="HWC")  # Shape (H, W, C)
            else:
                raise TypeError(f"Inst should be tensor or list. Got instead {type(inst)}.")
            output["hv_map"] = hv_map

        if self.labels:
            output["label"] = file
        return output

    def __len__(self) -> int:
        """
        Returns the dataset size
        """
        return len(self.data)

    @staticmethod
    def select_data(dataset: str) -> List[str]:
        """
        Splits the MoNuSeg dataset

        Parameters:
          dataset: "Train Kaggle", "Test Kaggle", "Train", "Test", "Surplus" or "Whole"
        """
        if dataset == "Train Kaggle":
            data = MoNuSeg.train + MoNuSeg.test
        elif dataset == "Test Kaggle":
            data = MoNuSeg.test_kaggle
        elif dataset == "Train":
            data = MoNuSeg.train
        elif dataset == "Test":
            data = MoNuSeg.test
        elif dataset == "Whole":
            data = MoNuSeg.train + MoNuSeg.test + MoNuSeg.test_kaggle
        elif dataset == "Surplus":
            data = MoNuSeg.surplus
        else:
            raise ValueError(f"Dataset should be of value 'Train Kaggle', 'Test Kaggle', Train', 'Test', 'Surplus' or "
                             f"'Whole'. Got instead {dataset}.")
        return data

    def _get_item_original(self, file: str, output: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieves the image and associated ground truth(s) in the original size (i.e., 1000x1000 pixels)."""
        img_file = Path(self.img_dir, file + ".tif")
        img = Image.open(img_file)
        output["img"] = img

        if self.segmentation_mask:
            seg_mask_file = Path(self.seg_mask_dir, file + ".npy")
            seg_mask = np.load(str(seg_mask_file))
            output["seg_mask"] = seg_mask
        if self.contour_mask:
            cont_mask_file = Path(self.cont_mask_dir, file + ".npy")
            cont_mask = np.load(str(cont_mask_file))
            output["cont_mask"] = cont_mask
        if self.distance_map:
            dist_map_file = Path(self.dist_map_dir, file + ".npy")
            dist_map = np.load(str(dist_map_file))
            output["dist_map"] = dist_map
        if self.instances:
            inst_file = Path(self.inst_dir, file + ".xml")
            inst = NucleiInstances.from_MoNuSeg(inst_file).as_ndarray()
            output["inst"] = inst

        return output

    def _get_item_256(self, file: str, output: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieves the image and associated ground truth(s) in the size 256x256 pixels."""
        img_file = Path(self.img_dir, file + ".pt")
        img = torch.load(img_file)
        output["img"] = img

        if self.segmentation_mask:
            seg_mask_file = Path(self.seg_mask_dir, file + ".pt")
            seg_mask = torch.load(seg_mask_file)
            output["seg_mask"] = seg_mask
        if self.contour_mask:
            cont_mask_file = Path(self.cont_mask_dir, file + ".pt")
            cont_mask = torch.load(cont_mask_file)
            output["cont_mask"] = cont_mask
        if self.distance_map:
            dist_map_file = Path(self.dist_map_dir, file + ".pt")
            dist_map = torch.load(dist_map_file)
            output["dist_map"] = dist_map
        if self.instances:
            inst_file = Path(self.inst_dir, file + ".pt")
            inst = torch.load(inst_file)
            output["inst"] = inst

        return output
