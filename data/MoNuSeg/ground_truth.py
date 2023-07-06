from pathlib import Path
from typing import List, Optional, Union
from xml.dom import minidom

import numpy as np
import torch
from scipy.ndimage import center_of_mass, distance_transform_cdt
from skimage.draw import polygon2mask
from skimage.morphology import binary_dilation, square, remove_small_objects
from skimage.segmentation import find_boundaries, flood
from torch import Tensor

from data.MoNuSeg.utils import get_bbox, prob_to_mask, cuda_tensor_to_ndarray
from postprocessing.graham_postprocessing import graham_postprocess
from postprocessing.naylor_postprocessing import naylor_postprocess


class NucleiInstances:
    SHAPE_KUMAR = (1000, 1000)

    def __init__(self, nuclei_instances: List[np.ndarray]):
        if nuclei_instances:
            self.shape = nuclei_instances[0].shape
        else:
            self.shape = None
        self.nuc_inst = nuclei_instances

    def __len__(self) -> int:
        return len(self.nuc_inst)

    def __getitem__(self, idx) -> np.ndarray:
        return self.nuc_inst[idx]

    def as_ndarray(self) -> np.ndarray:
        """Converts the nuclei instances as List[np.ndarray] into a np.ndarray of shape (H, W, #nuclei)."""
        array = np.array(self.nuc_inst)
        if self.nuc_inst:
            return np.transpose(array, (1, 2, 0))
        else:  # No nuclei instances
            return array

    def as_tensor(self) -> Tensor:
        """Converts the nuclei instances as List[np.ndarray] into a tensor of shape (#nuclei, H, W)."""
        array = np.array(self.nuc_inst)
        return torch.from_numpy(array)

    @staticmethod
    def from_MoNuSeg(label: Path) -> "NucleiInstances":
        """
        Extracts the nuclei instances from a xml-file.
        """

        dom = minidom.parse(str(label))
        tree = dom.documentElement
        regions = tree.getElementsByTagName("Region")
        nuc_inst = []
        empty_regions = 0
        for region in regions:
            vertexes = region.getElementsByTagName("Vertex")
            polygon = []
            for vertex in vertexes:
                polygon.append((float(vertex.getAttribute("Y")), float(vertex.getAttribute("X"))))
            mask = polygon2mask(image_shape=NucleiInstances.SHAPE_KUMAR, polygon=polygon)
            if mask.any():  # Check for emtpy masks
                nuc_inst.append(mask)
            else:
                empty_regions += 1
        print(f"Found {empty_regions} empty regions in {str(label)}")
        return NucleiInstances(nuc_inst)

    @staticmethod
    def from_seg(seg: Tensor, cont: Optional[Tensor] = None, seg_thresh: Union[str, float] = 0.5,
                 cont_thresh: Union[str, float] = 0.5) -> "NucleiInstances":
        """
        Extracts nuclei instances from a segmentation via flooding.

        If a contour is provided, then the contour is subtracted from the segmentation prior to flooding.
        """
        seg = cuda_tensor_to_ndarray(seg)
        seg_mask = prob_to_mask(seg, thresh=seg_thresh)
        remove_small_objects(seg_mask, min_size=15, out=seg_mask)  # Noise suppression; hard coded # Maybe insert later
        if cont is not None:
            cont = cuda_tensor_to_ndarray(cont)
            cont_mask = prob_to_mask(cont, cont_thresh)
            mask = seg_mask * np.logical_not(cont_mask)
        else:
            mask = seg_mask
        # Noise suppression:
        # remove_small_objects(mask, min_size=30, out=mask)

        height, width = mask.shape
        nuclei = []
        for y in range(height):
            for x in range(width):
                if mask[y, x]:
                    nucleus = flood(mask, seed_point=(y, x))  # Region growing with 8-connected neighborhood
                    mask = np.logical_xor(mask, nucleus)  # Removes the nucleus from the mask
                    # if cont is not None:
                    #     nucleus = binary_dilation(nucleus, square(3))  # * seg_mask
                    nuclei.append(nucleus)
        if cont is not None:
            flag = True
            while flag:
                flag = False
                for idx, nucleus in enumerate(nuclei):
                    a = binary_dilation(nucleus, footprint=square(3, dtype=bool), out=None) * cont_mask
                    if a.any():
                        nuclei[idx] += a
                        cont_mask = cont_mask * ~a
                        flag = True
        return NucleiInstances(nuclei)

    @staticmethod
    def from_dist_map(dist_map: Tensor, param: int, thresh: Union[int, float]) -> "NucleiInstances":
        """
        Extracts nuclei instances from a distance map via the postprocessing strategy by Naylor et al. 2019.
        """

        dist_map[dist_map > 255] = 255
        dist_map = dist_map.byte()

        dist_map = cuda_tensor_to_ndarray(dist_map)

        labeled_inst = naylor_postprocess(dist_map, param, thresh)
        return NucleiInstances.from_labeled_inst(labeled_inst)

    @staticmethod
    def from_hv_map(hv_map: Tensor, seg: Tensor, seg_thresh: Union[str, float] = 0.5) -> "NucleiInstances":
        """
        Extracts nuclei instances from a horizontal and vertical distance map via the postprocessing strategy by
         Graham et al. 2019.
        """

        hv_map = cuda_tensor_to_ndarray(hv_map)

        seg = cuda_tensor_to_ndarray(seg)
        seg_mask = prob_to_mask(seg, seg_thresh)

        labeled_inst = graham_postprocess(hv_map, seg_mask)

        return NucleiInstances.from_labeled_inst(labeled_inst)

    @staticmethod
    def from_labeled_inst(labeled_inst: np.ndarray) -> "NucleiInstances":
        """
        Extracts nuclei instances from a map of labeled instances.

        Premiss: Labels are continuous (e.g. [0, 1, 2, 3] and not [0, 1, 3]).
        """
        nuclei = []
        for l in range(1, labeled_inst.max()+1):  # 0 is background
            nucleus = labeled_inst == l
            nuclei.append(nucleus)
        return NucleiInstances(nuclei)

    @staticmethod
    def from_inst(inst: Tensor) -> "NucleiInstances":
        """
        Extracts nuclei instances as list of arrays from a stacked tensor.
        """
        inst = inst.cpu()
        inst = inst.numpy()  # Conversion to np.ndarray of shape (C, H, W)
        nuclei = [nucleus for nucleus in inst]
        return NucleiInstances(nuclei)

    def to_labeled_inst(self) -> np.ndarray:
        """
        Generates a mask of labeled instances from the nuclei instances.

        Caution: Causes information loss if nuclei instances overlap!
        """
        labels = np.zeros(self.shape, dtype=int)
        for i, inst in enumerate(self.nuc_inst, start=1):
            labels[inst] = i
        return labels

    def to_seg_mask(self) -> np.ndarray:
        """
        Generates a binary mask from the nuclei instances.
        """
        mask = np.zeros(shape=self.shape, dtype=bool)
        for inst in self.nuc_inst:
            mask += inst
        return mask.astype(np.float32)

    def to_cont_mask(self) -> np.ndarray:
        """
        Generates a binary mask of the nuclei contours from the nuclei instances.
        """
        mask = np.zeros(shape=self.shape, dtype=bool)
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
        return mask.astype(np.float32)

    def to_dist_map(self) -> np.ndarray:
        """
        Generates a distance map from the nuclei instances.

        Caution: Causes information loss if nuclei instances overlap!
        """
        map = np.zeros(shape=self.shape, dtype=np.float32)
        for inst in self.nuc_inst:
            dist = distance_transform_cdt(inst, metric="chessboard")
            # if map[inst].any():  # Checks for overlap with another nucleus
            #     [inst] = np.mean((map[inst], dist[inst]), axis=2)
            #     # map[inst] = (map[inst] + dist[inst]) / 2
            # else:
            #     map += dist
            map[inst] = dist[inst]
        return map

    def to_hv_map(self, order: str = "CHW") -> np.ndarray:
        """
        Generates a horizontal and a vertical distance map form the nuclei instances.
        """
        h_map = np.zeros(shape=self.shape, dtype=np.float32)
        v_map = np.zeros(shape=self.shape, dtype=np.float32)
        for inst in self.nuc_inst:
            # Determine bounding box (bbox) for the nucleus:
            y_min, y_max, x_min, x_max = get_bbox(inst)
            crop = inst[y_min:y_max, x_min:x_max]

            # Calculate nucleus center as
            # a) the center of mass or
            cntr = center_of_mass(crop)
            cntr = int(cntr[0] + 0.5), int(cntr[1] + 0.5)
            # b) the geometric center
            # cntr = (crop.shape[0] // 2, crop.shape[1] // 2)

            # Calculate centered row and column values:
            row = np.arange(crop.shape[1], dtype=np.float32) - cntr[1]
            col = np.arange(crop.shape[0], dtype=np.float32) - cntr[0]
            # Normalize to [-1, 1]:
            row[row < 0] /= -row[0]
            row[row > 0] /= row[-1]
            col[col < 0] /= -col[0]
            col[col > 0] /= col[-1]

            # Insert horizontal and vertical distance map:
            h_map[y_min:y_max, x_min:x_max] = np.where(crop, row, h_map[y_min:y_max, x_min:x_max])
            v_map[y_min:y_max, x_min:x_max] = np.where(crop.T, col, v_map[y_min:y_max, x_min:x_max].T).T

        # Return h_map, v_map
        if order == "CHW":
            return np.stack((h_map, v_map))  # Shape (C, H, W)
        elif order == "HWC":
            return np.dstack((h_map, v_map))  # Shape (H, W, C)
        else:
            raise ValueError(f"Order should be CHW or HWC. Got instead {order}")


def setup():
    from data.MoNuSeg.data_module import MoNuSegDataModule

    data_module = MoNuSegDataModule(
        seg_masks=True,
        cont_masks=True,
        dist_maps=True,
        labels=False,
        data_root="../../datasets"
    )
    data_module.setup(stage="test")
    test_loader = data_module.test_dataloader()
    img, seg_mask, cont_mask, dist_map, inst_gt = next(iter(test_loader))
    inst = inst_gt[0].numpy()
    return inst


if __name__ == "__main__":
    from data.MoNuSeg.illustrator import Picture

    # res = timeit(setup="from data.MoNuSeg.ground_truth import NucleiInstances, setup; inst=setup()[0]",
    #              stmt="NucleiInstances([inst]).to_hv_map()", number=int(1e5))
    # print(res)

    inst = setup()
    hv_map = NucleiInstances([inst[0], inst[1]]).to_hv_map()
    print(hv_map.shape)
    Picture(hv_map[0]).show()
    Picture(hv_map[1]).show()
    Picture(inst[0]).show()
    Picture(inst[1]).show()

    # y_min, y_max, x_min, x_max = get_bbox(inst[0])
    # Picture(inst[0][y_min:y_max, x_min:x_max]).show()
