import cv2
import numpy as np
from scipy.ndimage import label
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects


def graham_postprocess(hv_map: np.ndarray, seg_mask: np.ndarray) -> np.ndarray:
    """
    Postprocessing pipeline by Graham et al. 2019.

    Code taken from:
    https://github.com/vqdang/hover_net/blob/master/models/hovernet/post_proc.py
    """
    # Noise suppression:
    remove_small_objects(seg_mask, min_size=10, out=seg_mask)  # Hard coded size
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
    contours = np.array(combined >= 0.4, dtype=np.int32)  # Hard coded threshold

    # Remove contours from nuclei:
    marker = seg_mask - contours
    marker[marker < 0] = 0
    ## Fill holes:
    # marker = binary_fill_holes(marker).astype("uint8")
    ## Morphological opening:
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    ## Label marker:
    # se = np.array([[0,1,0],[1,1,1],[0,1,0]])  # 1-connected neighborhood
    se = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])  # 2-connected neighborhood
    marker = label(marker, structure=se)[0]
    ## Remove small objects:
    # marker = remove_small_objects(marker, min_size=4)  # Hard coded size

    return watershed(energy_landscape, markers=marker, mask=seg_mask)