import numpy as np
from typing import Any, List, NoReturn, Union
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

from data.MoNuSeg.ground_truth import NucleiInstances


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

    def __init__(self, root: str = "datasets", segmentation_mask: bool = True, contour_mask: bool = True,
                 distance_map: bool = True, instances: bool = False, transforms=None, dataset: Union[List[str], str] = "Whole") -> NoReturn:
        self.segmentation_mask = segmentation_mask
        self.contour_mask = contour_mask
        self.distance_map = distance_map
        self.instances = instances
        self.transforms = transforms

        if self.instances and self.transforms is not None:
            print("Please note: Transforms are not applied to the nuclei instances!")

        base_dir = Path(root, "MoNuSeg 2018")
        self.img_dir = Path(base_dir, "Tissue Images")
        self.inst_dir = Path(base_dir, "Annotations")
        self.seg_mask_dir = Path(base_dir, "Segmentation masks")
        self.cont_mask_dir = Path(base_dir, "Contour masks")
        self.dist_map_dir = Path(base_dir, "Distance maps")
        if isinstance(dataset, str):
            self.data = MoNuSeg.select_data(dataset)
        elif isinstance(dataset, list):
            self.data = dataset
        else:
            raise TypeError(f"Dataset should be of type list or str. Got instead {type(dataset)}.")

    def __getitem__(self, idx: int) -> List[Any]:
        """
        Retrieves the image and the associated ground truth(s)
        """
        file = self.data[idx]
        output = []

        img_file = Path(self.img_dir, file + ".tif")
        img = Image.open(img_file)
        output.append(img)

        if self.segmentation_mask:
            seg_mask_file = Path(self.seg_mask_dir, file + ".npy")
            seg_mask = np.load(str(seg_mask_file))
            output.append(seg_mask)
        if self.contour_mask:
            cont_mask_file = Path(self.cont_mask_dir, file + ".npy")
            cont_mask = np.load(str(cont_mask_file))
            output.append(cont_mask)
        if self.distance_map:
            dist_map_file = Path(self.dist_map_dir, file + ".npy")
            dist_map = np.load(str(dist_map_file))
            output.append(dist_map)

        if self.transforms is not None:
            output = self.transforms(output)

        if self.instances:
            inst_file = Path(self.inst_dir, file + ".xml")
            inst = NucleiInstances.from_MoNuSeg(inst_file).nuc_inst
            output.append(inst)

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
            raise ValueError(f"Dataset should be of value 'Train Kaggle', 'Test Kaggle', Train', 'Test', 'Surplus', 'Whole'. Got instead {dataset}.")
        return data
