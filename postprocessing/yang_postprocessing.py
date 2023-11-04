from typing import List, Optional, Tuple

import numpy as np
from skimage.morphology import binary_erosion, label, remove_small_objects
from skimage.segmentation import watershed


def yang_postprocess(seg_mask: np.ndarray, yang_params: Optional[dict] = None) -> np.ndarray:
    """
    Postprocessing pipline from Yang et al. 2006.

    Nuclei instances are extracted using the watershed algorithm. The Markers for watershed segmentation are obtained
    by conditional erosion of the (nuclei) segmentation mask: First, the mask is iteratively eroded with coarse erosion
    structures until the object size is smaller than the threshold thresh_coarse. Second, the mask is further
    iteratively eroded with fine erosion structures until the object size is smaller than the threshold thresh_fine.

    Please consult for more algorithmic details the original publication "Nuclei segmentation using marker-controlled
    watershed, tracking using mean-shift, and Kalman filter in time-lapse microscopy" by Yang et al. 2006.
    """

    def classify_objects_by_size(mask: np.ndarray, thresh: int) -> Tuple[np.ndarray, np.ndarray]:
        labeled_objects, num_objects = label(mask, connectivity=2, return_num=True)  # Two-connected neighborhood

        object_sizes = np.bincount(labeled_objects.ravel())[1:]  # Ignore background count at zero index
        object_indices = np.arange(1, num_objects + 1)  # Background correspond to 0
        objects_greater_or_equal = np.isin(labeled_objects, object_indices[object_sizes >= thresh])
        objects_smaller = mask * ~objects_greater_or_equal

        return objects_greater_or_equal, objects_smaller

    def conditional_erosion(mask: np.ndarray, thresh: int, erosion_structures: List[np.ndarray]) -> np.ndarray:
        idx = 0
        to_be_eroded, output = classify_objects_by_size(mask, thresh=thresh)
        while to_be_eroded.any():
            structure = erosion_structures[idx]
            eroded = binary_erosion(to_be_eroded, footprint=structure)
            idx = (idx + 1) % len(erosion_structures)
            to_be_eroded, not_to_be_eroded = classify_objects_by_size(eroded, thresh=thresh)
            output += not_to_be_eroded
        return output

    struc_fine = [
        np.array([[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]], dtype=np.uint8),
        np.array([[1, 0, 1],
                  [0, 1, 0],
                  [1, 0, 1]], dtype=np.uint8),
    ]
    struc_coarse = [
        np.array([[0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1, 0, 0],
                  [0, 1, 1, 1, 1, 1, 0],
                  [1, 1, 1, 1, 1, 1, 1],
                  [0, 1, 1, 1, 1, 1, 0],
                  [0, 0, 1, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8),
        np.array([[0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 1, 1, 1, 0, 0],
                  [0, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 0],
                  [0, 0, 1, 1, 1, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0]], dtype=np.uint8),
        np.array([[1, 0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 0, 0, 0],
                  [0, 1, 1, 1, 1, 0, 0],
                  [0, 1, 1, 1, 1, 1, 0],
                  [0, 0, 1, 1, 1, 1, 0],
                  [0, 0, 0, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 1]], dtype=np.uint8),
        np.array([[0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 1, 1, 1, 0],
                  [0, 0, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 0, 0],
                  [0, 1, 1, 1, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0, 0]], dtype=np.uint8),
    ]

    # Define default hyperparameters:
    default_params = {
        "min_obj_size": 10,
        "thresh_coarse": 310,
        "thresh_fine": 240
    }
    if yang_params is None:
        yang_params = {}
    # Overwrite default hyperparameters:
    yang_params = {**default_params, **yang_params}

    remove_small_objects(seg_mask, min_size=yang_params["min_obj_size"], out=seg_mask)
    mask = seg_mask.copy()

    mask = conditional_erosion(mask, thresh=yang_params["thresh_coarse"], erosion_structures=struc_coarse)
    mask = conditional_erosion(mask, thresh=yang_params["thresh_fine"], erosion_structures=struc_fine)
    marker = label(mask, connectivity=1, background=0)

    return watershed(seg_mask, markers=marker, mask=seg_mask)  # Labeled instances
