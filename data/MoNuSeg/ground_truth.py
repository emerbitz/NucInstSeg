from pathlib import Path
from typing import List
from xml.dom import minidom

import numpy as np
import torch
from scipy.ndimage import distance_transform_cdt
from skimage.draw import polygon2mask
from skimage.segmentation import find_boundaries, flood
from torch import Tensor


class NucleiInstances:

    def __init__(self, nuclei_instances: List[np.ndarray]) -> None:
        self.nuc_inst = nuclei_instances

    def __len__(self) -> int:
        return len(self.nuc_inst)

    def __getitem__(self, idx) -> np.ndarray:
        return self.nuc_inst[idx]

    def as_ndarray(self) -> np.ndarray:
        """Converts the nuclei instances as List[np.ndarray] into a np.ndarray of shape (H, W, #nuclei)"""
        array = np.array(self.nuc_inst)
        return np.transpose(array, (1, 2, 0))

    def as_tensor(self) -> Tensor:
        """Converts the nuclei instances as List[np.ndarray] into a torch.Tensor of shape (#nuclei, H, W)"""
        array = np.array(self.nuc_inst)
        return torch.from_numpy(array)

    @staticmethod
    def from_MoNuSeg(label: Path) -> "NucleiInstances":
        """
        Extracts the nuclei instances from a xml-file
        """

        img_shape = (1000, 1000)
        dom = minidom.parse(str(label))
        tree = dom.documentElement
        regions = tree.getElementsByTagName("Region")
        nuc_inst = []
        for region in regions:
            vertexes = region.getElementsByTagName("Vertex")
            polygon = []
            for vertex in vertexes:
                polygon.append((float(vertex.getAttribute("Y")), float(vertex.getAttribute("X"))))
            nuc_inst.append(polygon2mask(image_shape=img_shape, polygon=polygon))
        return NucleiInstances(nuc_inst)

    @staticmethod
    def from_seg_mask(seg_mask: torch.Tensor) -> "NucleiInstances":
        """Extracts nuclei instances from a segmentation mask"""

        seg_mask = seg_mask.squeeze().numpy()  # Conversion to np.ndarray of shape (H, W)
        height, width = seg_mask.shape
        nuclei = []
        for y in range(height):
            for x in range(width):
                if seg_mask[y, x]:
                    nucleus = flood(seg_mask, seed_point=(y, x))  # Region growing with 8-connected neighborhood
                    seg_mask = np.logical_xor(seg_mask, nucleus)  # Removes the nucleus from the segmentation mask
                    nuclei.append(nucleus)
        return NucleiInstances(nuclei)

    def to_seg_mask(self) -> np.ndarray:
        """
        Generates a binary mask from the nuclei instances
        """

        mask_shape = self.nuc_inst[0].shape
        mask = np.zeros(shape=mask_shape, dtype=bool)
        for inst in self.nuc_inst:
            mask *= inst
            # mask = np.logical_or(mask, inst)
        return mask.astype(float)

    def to_cont_mask(self) -> np.ndarray:
        """
        Generates a binary mask of the nuclei contours from the nuclei instances
        """

        mask_shape = self.nuc_inst[0].shape
        mask = np.zeros(shape=mask_shape, dtype=bool)
        for inst in self.nuc_inst:
            contours = find_boundaries(
                inst,
                connectivity=2,
                mode="inner",
                background=False
            )  # Connectivity=1: 4-connected neighborhood, Connectivity=2: 8-connected neighborhood
            # Find_boundaries(mode="tick") = find_boundaries(mode="inner") logical_or find_boundaries(mode="outer")
            mask += contours
            # mask = np.logical_or(mask, contours)
        return mask.astype(float)

    def to_dist_map(self) -> np.ndarray:
        """
        Generates a distance map from the nuclei instances
        """

        map_shape = self.nuc_inst[0].shape
        map = np.zeros(shape=map_shape, dtype=float)
        for inst in self.nuc_inst:
            dist = distance_transform_cdt(inst, metric="chessboard")
            # if map[inst].any():  # Checks for overlap with another nucleus
            #     [inst] = np.mean((map[inst], dist[inst]), axis=2)
            #     # map[inst] = (map[inst] + dist[inst]) / 2
            # else:
            #     map += dist
            map[inst] = dist[inst]
        return map
