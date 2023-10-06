from typing import Optional

import numpy as np
from skimage.morphology import label, remove_small_objects
from skimage.segmentation import watershed


def noname_postprocess(seg_mask: np.ndarray, cont_mask: np.ndarray, noname_params: Optional[dict] = None) -> np.ndarray:
    """
    Noname postprocessing pipeline for nuclei instance extraction.

    Nuclei instances are extracted using the watershed algorithm. Markers for watershed segmentation are obtained
    by removing the nuclei contours from the whole nuclei.
    """
    # Define default hyperparameters:
    default_params = {
        "min_obj_size": 65,
        "min_marker_size": 75
    }
    if noname_params is None:
        noname_params = {}
    # Overwrite default hyperparameters:
    noname_params = {**default_params, **noname_params}

    # Noise suppression:
    remove_small_objects(seg_mask, min_size=noname_params["min_obj_size"], out=seg_mask, connectivity=1)
    # Remove nuclei contours from nuclei segmentation:
    marker = seg_mask * ~cont_mask
    # Remove artifacts due to nuclei overlap:
    remove_small_objects(marker, min_size=noname_params["min_marker_size"], out=marker, connectivity=1)
    # Add small nuclei removed in the previous step:
    marker += ~watershed(seg_mask, markers=marker, mask=seg_mask, connectivity=1).astype(bool) * seg_mask
    # Watershed:
    return watershed(seg_mask, markers=label(marker), mask=seg_mask, connectivity=1)
