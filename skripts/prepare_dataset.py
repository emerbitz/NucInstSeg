from typing import Union, Tuple

from data.MoNuSeg.dataset import MoNuSeg
from data.MoNuSeg.dataset_creator import MoNuSegCreator
from data.MoNuSeg.dataset_patcher import MoNuSegPatcher
from transformation.transformations import ToTensor


def prepare_data(data_split: str = "Train Kaggle", data_root: str = "datasets",
                 seg_masks: bool = True, cont_masks: bool = True, dist_maps: bool = True,
                 img_size: Union[Tuple[int, int], int] = (256, 256)
                 ):
    """
    Prepares the MoNuSeg dataset prior to usage.

    First, the ground truth representations (i.e., the classification probability and distance maps) are generated from
    the nuclei instances. Subsequently, image patches are extracted from each whole slide image (WSI) and its
    corresponding ground truth.
    """
    # Create ground truth representations
    creator = MoNuSegCreator(root=data_root)
    creator.save_ground_truths(
        segmentation_masks=seg_masks,
        contour_masks=cont_masks,
        distance_maps=dist_maps
    )
    # Cut
    patcher = MoNuSegPatcher(
        dataset=MoNuSeg(
            root=data_root,
            segmentation_masks=seg_masks,
            contour_masks=cont_masks,
            distance_maps=dist_maps,
            instances=True,
            transforms=ToTensor(),
            dataset=data_split
        )
    )
    patcher.split_and_save(patch_size=img_size)


def main():
    prepare_data(
        seg_masks=True,
        cont_masks=True,
        dist_maps=True,
        img_size=(256, 256),
        data_root="../datasets"
    )


"""
This skript needs to be executed prior to the usage of the MoNuSeg dataset.

The skript generates the required image patches of each whole slide image (WSI) and the corresponding ground truth.
"""

if __name__ == "__main__":
    main()