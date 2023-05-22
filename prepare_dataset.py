from typing import Union, Tuple

from data.MoNuSeg.dataset import MoNuSeg
from data.MoNuSeg.dataset_creator import MoNuSegCreator
from data.MoNuSeg.dataset_patcher import MoNuSegPatcher
from transformation.transformations import ToTensor


def prepare_data(data_split: str = "Train Kaggle", data_root: str = "datasets",
                 seg_masks: bool = True, cont_masks: bool = True, dist_maps: bool = True, hv_maps: bool = True,
                 img_size: Union[Tuple[int, int], int] = (256, 256)
                 ):
    # Create ground truths
    creator = MoNuSegCreator(root=data_root)
    creator.save_ground_truths(
        segmentation_masks=seg_masks,
        contour_masks=cont_masks,
        distance_maps=dist_maps,
        hv_distance_maps=hv_maps
    )
    # Cut
    patcher = MoNuSegPatcher(
        dataset=MoNuSeg(
            root=data_root,
            segmentation_masks=seg_masks,
            contour_masks=cont_masks,
            distance_maps=dist_maps,
            hv_distance_maps=hv_maps,
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
        hv_maps=True,
        img_size=(256, 256),
        data_root="datasets"
    )


if __name__ == "__main__":
    main()