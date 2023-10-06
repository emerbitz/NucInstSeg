from typing import Optional
import warnings

import cv2
import numpy as np
from scipy.ndimage import label
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed

from data.MoNuSeg.utils import threshold


def exprmtl_postprocess(hv_map: np.ndarray, seg_mask: np.ndarray, exprmtl_params: Optional[dict] = None) -> np.ndarray:
    """
    Experimental postprocessing pipeline for horizontal and vertical distance maps.

    Based on the code from Graham et al. 2019:
    https://github.com/vqdang/hover_net/blob/master/models/hovernet/post_proc.py
    """
    # Define default hyperparameters:
    default_params = {
        "min_obj_size": 10,
        "thresh_comb": "triangle",
        "min_marker_size": 10
    }
    if exprmtl_params is None:
        exprmtl_params = {}
    # Overwrite default hyperparameters:
    exprmtl_params = {**default_params, **exprmtl_params}

    # Noise suppression:
    remove_small_objects(seg_mask, min_size=exprmtl_params["min_obj_size"], out=seg_mask)
    # Min-max normalization:
    h_map = cv2.normalize(hv_map[0], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    v_map = cv2.normalize(hv_map[1], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # Sobel filtering for edge detection:
    sobel_h = cv2.Sobel(h_map, cv2.CV_64F, 1, 0, ksize=21)
    sobel_v = cv2.Sobel(v_map, cv2.CV_64F, 0, 1, ksize=21)
    # Min-max normalization:
    sobel_h_norm = cv2.normalize(sobel_h, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    sobel_v_norm = cv2.normalize(sobel_v, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # Combine vertical and horizontal distance maps:
    combined = np.minimum(sobel_h_norm, sobel_v_norm)

    # Thresholding to generate marker:
    marker = threshold(combined, thresh=exprmtl_params["thresh_comb"])
    # Remove background:
    marker = marker - (1 - seg_mask)
    marker[marker < 0] = 0

    # Label marker:
    se = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])  # 2-connected neighborhood
    marker = label(marker, structure=se)[0]

    # Remove small objects:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        marker = remove_small_objects(marker, min_size=exprmtl_params["min_marker_size"])  # Hard coded size

    return watershed(seg_mask, markers=marker, mask=seg_mask)
    # Maybe usage of raw segmentation or background distance (i.e., abs(h + v)) as image
