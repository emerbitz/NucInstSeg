from typing import Optional

import numpy as np
from skimage.morphology import label, remove_small_objects


def baseline_postprocess(seg_mask: np.ndarray, baseline_params: Optional[dict] = None) -> np.ndarray:
    """
    Postprocessing pipeline serving as baseline for nuclei instance extraction.

    Nuclei instances are obtained by labeling of the (nuclei) segmentation mask. Thus, touching/overlapping nuclei are
    not separated. This pipeline is solely intended as a baseline for performance comparison with the other
    postprocessing pipelines and not expected to do great.
    """
    # Define default hyperparameters:
    default_params = {
        "min_obj_size": 50,
    }
    if baseline_params is None:
        baseline_params = {}
    # Overwrite default hyperparameters:
    baseline_params = {**default_params, **baseline_params}

    # Noise suppression:
    remove_small_objects(seg_mask, min_size=baseline_params["min_obj_size"], out=seg_mask, connectivity=1)
    return label(seg_mask, connectivity=1)
