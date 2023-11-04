from typing import Optional, Union

import numpy as np
from skimage import img_as_ubyte
from skimage.measure import label
from skimage.morphology import reconstruction, dilation, erosion, disk, square
from skimage.segmentation import watershed

from data.MoNuSeg.utils import threshold


def naylor_postprocess(dist_map: np.ndarray, naylor_params: Optional[dict] = None, param: int = 7,
                       thresh: Union[int, float] = 0.5) -> np.ndarray:
    """
    Postprocessing pipeline based on dynamic watershed segmentation by Naylor et al. 2019.

    Code taken from:
    https://github.com/PeterJackNaylor/DRFNS/blob/73fc5683db5e9f860846e22c8c0daf73b7103082/src_RealData/postproc/postprocessing.py
    """
    # Define default hyperparameters:
    default_params = {
        "thresh_dist": "triangle",
        "h_param": 1,
    }
    if naylor_params is None:
        naylor_params = {}
    # Overwrite default hyperparameters:
    naylor_params = {**default_params, **naylor_params}

    return DynamicWatershedAlias(dist_map, naylor_params["h_param"], naylor_params["thresh_dist"])


"""
Code hereafter is copied-and-pasted from the GitHub repository of Peter Naylor:
https://github.com/PeterJackNaylor/DRFNS/blob/73fc5683db5e9f860846e22c8c0daf73b7103082/src_RealData/postproc/postprocessing.py

Adaption:
* Threshold for distance map can now also be automatically determined.
"""

def PrepareProb(img, convertuint8=True, inverse=True):
    """
    Prepares the prob image for post-processing, it can convert from
    float -> to uint8 and it can inverse it if needed.
    """
    if convertuint8:
        img = img_as_ubyte(img)
    if inverse:
        img = 255 - img
    return img


def HreconstructionErosion(prob_img, h):
    """
    Performs a H minimma reconstruction via an erosion method.
    """

    def making_top_mask(x, lamb=h):
        return min(255, x + lamb)

    f = np.vectorize(making_top_mask)
    shift_prob_img = f(prob_img)

    seed = shift_prob_img
    mask = prob_img
    recons = reconstruction(
        seed, mask, method='erosion').astype(np.dtype('ubyte'))
    return recons


def find_maxima(img, convertuint8=False, inverse=False, mask=None):
    """
    Finds all local maxima from 2D image.
    """
    img = PrepareProb(img, convertuint8=convertuint8, inverse=inverse)
    recons = HreconstructionErosion(img, 1)
    if mask is None:
        return recons - img
    else:
        res = recons - img
        res[mask == 0] = 0
        return res


def GetContours(img):
    """
    Returns only the contours of the image.
    The image has to be a binary image
    """
    img[img > 0] = 1
    return dilation(img, disk(2)) - erosion(img, disk(2))


def generate_wsl(ws):
    """
    Generates watershed line that correspond to areas of
    touching objects.
    """
    se = square(3)
    ero = ws.copy()
    ero[ero == 0] = ero.max() + 1
    ero = erosion(ero, se)
    ero[ws == 0] = 0

    grad = dilation(ws, se) - ero
    grad[ws == 0] = 0
    grad[grad > 0] = 255
    grad = grad.astype(np.uint8)
    return grad


def ArrangeLabel(mat):
    """
    Arrange label image as to effectively put background to 0.
    """
    val, counts = np.unique(mat, return_counts=True)
    background_val = val[np.argmax(counts)]
    mat = label(mat, background=background_val)
    if np.min(mat) < 0:
        mat += np.min(mat)
        mat = ArrangeLabel(mat)
    return mat


def DynamicWatershedAlias(p_img, lamb=7, p_thresh=0.5):
    """
    Applies our dynamic watershed to 2D prob/dist image.
    """
    # b_img = (p_img > p_thresh) + 0
    b_img = threshold(p_img, p_thresh).astype(np.int32)  # Relation: greater_or_equal instead of greater (cf. above)
    Probs_inv = PrepareProb(p_img)

    Hrecons = HreconstructionErosion(Probs_inv, lamb)
    markers_Probs_inv = find_maxima(Hrecons, mask=b_img)
    markers_Probs_inv = label(markers_Probs_inv)
    ws_labels = watershed(Hrecons, markers_Probs_inv, mask=b_img)
    arrange_label = ArrangeLabel(ws_labels)
    wsl = generate_wsl(arrange_label)
    arrange_label[wsl > 0] = 0

    return arrange_label
