from typing import Optional, Tuple, Union
from torch import Tensor

from data.MoNuSeg.ground_truth import NucleiInstances
from postprocessing.postprocesses_base import Postprocess
from data.MoNuSeg.ground_truth import NucleiInstances

class SegPostProcess(Postprocess):
    """
    Postprocessing based on the segmentation mask and the contour mask.

    Nuclei instances are extracted by flooding. If a contour mask is provided, then the contour mask is subtracted from
    the segmentation mask prior to flooding.
    """
    def __init__(self):
        pass

    def __call__(self, seg: Tensor, cont: Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, ...]]:
        seg_mask = seg >= 0.5
        if cont is not None:
            cont_mask = cont >= 0.5
        else:
            cont_mask = cont
        return self.postprocess(seg_mask, cont_mask)

    def postprocess_fn(self, seg_mask: Tensor, cont_mask: Tensor = None) -> Tensor:
        return NucleiInstances.from_mask(seg_mask, cont_mask).as_tensor()


class DistPostProcess(Postprocess):
    """
    Postprocessing based on the distance map.

    The postprocessing strategy by Naylor et al. 2019 is used to identify nuclei instances.
    """
    def __init__(self, param: int, thresh: Union[int, float]):
        self.param = param
        self.thresh = thresh

    def __call__(self, dist: Tensor) -> Union[Tensor, Tuple[Tensor, ...]]:
        return self.postprocess(dist)

    def postprocess_fn(self, dist_map: Tensor, *args) -> Tensor:
        return NucleiInstances.from_dist_map(dist_map, self.param, self.thresh).as_tensor()


class HVPostProcess(Postprocess):
    """
   Postprocessing based on the horizontal and vertical distance map.

   The postprocessing strategy by Graham et al. 2019 is used to identify nuclei instances.
   """
    def __init__(self):
        pass

    def __call__(self, hv_map: Tensor, seg: Tensor) -> Union[Tensor, Tuple[Tensor, ...]]:
        seg_mask = seg >= 0.5
        return self.postprocess(hv_map, seg_mask)

    def postprocess_fn(self, hv_map: Tensor, seg_mask: Tensor) -> Tensor:
        return NucleiInstances.from_hv_map(hv_map, seg_mask).as_tensor()


if __name__ == '__main__':
    from data.MoNuSeg.data_module import MoNuSegDataModule
    from data.MoNuSeg.ground_truth import NucleiInstances
    from data.MoNuSeg.illustrator import Picture

    data_module = MoNuSegDataModule(
        seg_masks=True,
        cont_masks=True,
        dist_maps=True,
        hv_maps=False,
        labels=False,
        data_root="../datasets"
    )
    # data_module.prepare_data()
    data_module.setup(stage="test")
    test_loader = data_module.test_dataloader()
    img, seg_mask, cont_mask, dist_map, inst_gt = next(iter(test_loader))

    index = 7
    inst_pred = DistPostProcess(param=2, thresh=0)(dist_map)

    # pd, gt = inst_gt[index], inst_gt[index]
    # pd_labeled, gt_labeled = NucleiInstances.from_inst(pd).to_labeled_inst(), NucleiInstances.from_inst(gt).to_labeled_inst()
    # Picture(pd_labeled).show(), Picture(gt_labeled).show()

    print(inst_gt[index].shape, inst_pred[index].shape)
    # Picture.from_tensor(img[index]).show()
    Picture.from_tensor(img[index], inst_gt[index]).show()
    Picture.from_tensor(img[index], inst_pred[index]).show()

