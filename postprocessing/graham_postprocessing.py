import warnings
from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import label, binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed

from data.MoNuSeg.utils import threshold


def graham_postprocess(hv_map: np.ndarray, seg_mask: np.ndarray, graham_params: Optional[dict] = None) -> np.ndarray:
    """
    Postprocessing pipeline from Graham et al. 2019.

    Code taken from:
    https://github.com/vqdang/hover_net/blob/master/models/hovernet/post_proc.py

    Adaption:
    * Warnings of remove_small_objects are disabled, as the function throws a warning,
    if only a single object was detected.
    * Selected hyperparameters are no longer hard coded but can be specified in the dict 'graham_params'.
    """
    # Define default hyperparameters:
    default_params = {
        "min_obj_size": 10,
        "thresh_comb": 0.4,
        "min_marker_size": 10
    }
    if graham_params is None:
        graham_params = {}
    # Overwrite default hyperparameters:
    graham_params = {**default_params, **graham_params}

    # Noise suppression:
    remove_small_objects(seg_mask, min_size=graham_params["min_obj_size"], out=seg_mask)
    # Min-max normalization:
    h_map = cv2.normalize(hv_map[0], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    v_map = cv2.normalize(hv_map[1], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # Sobel filtering for edge detection:
    sobel_h = cv2.Sobel(h_map, cv2.CV_64F, 1, 0, ksize=21)
    sobel_v = cv2.Sobel(v_map, cv2.CV_64F, 0, 1, ksize=21)
    # Min-max normalization and inversion:
    sobel_h = 1 - cv2.normalize(sobel_h, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    sobel_v = 1 - cv2.normalize(sobel_v, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # Combine vertical and horizontal distance maps:
    combined = np.maximum(sobel_h, sobel_v)
    # Set background values to zero:
    combined = combined - (1 - seg_mask)
    combined[combined < 0] = 0

    # Calculate energy landscape:
    energy_landscape = (1.0 - combined) * seg_mask
    # Smoothing and inversion as preparation for watershed:
    energy_landscape = -cv2.GaussianBlur(energy_landscape, (3, 3), 0)
    # Thresholding to obtain contours:
    # contours = np.array(combined >= graham_params["thresh_comb"], dtype=np.int32)
    contours = threshold(combined, graham_params["thresh_comb"]).astype(dtype=np.int32)

    # Remove contours from nuclei:
    marker = seg_mask - contours
    marker[marker < 0] = 0
    # Fill holes:
    marker = binary_fill_holes(marker).astype("uint8")
    # Morphological opening:
    # marker = marker.astype("uint8")  # Uncomment if no filling is performed
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    # Label marker:
    se = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])  # 1-connected neighborhood
    # se = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])  # 2-connected neighborhood
    marker = label(marker, structure=se)[0]
    # Remove small objects:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        marker = remove_small_objects(marker, min_size=graham_params["min_marker_size"])

    return watershed(energy_landscape, markers=marker, mask=seg_mask)
