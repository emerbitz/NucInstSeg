from typing import Dict, Tuple, Union

import torch
from torch import Tensor

from postprocessing.postprocesses_base import Postprocess


class SegPostProcess(Postprocess):
    """
    Postprocessing based on the segmentation and the contour.

    Nuclei instances are extracted by flooding. If a contour is provided, then the contour is subtracted from
    the segmentation prior to flooding.
    """

    def __init__(self, seg_thresh: Union[str, float] = 0.5, cont_thresh: Union[str, float] = 0.5):
        self.seg_thresh = seg_thresh
        self.cont_thresh = cont_thresh

    def __call__(self, pred: Dict[str, Tensor]) -> Union[Tensor, Tuple[Tensor, ...]]:
        seg = pred["seg_mask"]
        cont = pred["cont_mask"]
        if torch.is_floating_point(seg):
            seg = torch.sigmoid(seg)
        if torch.is_floating_point(cont):
            cont = torch.sigmoid(cont)
        return self.postprocess(seg, cont)

    def postprocess_fn(self, seg: Tensor, cont: Tensor = None) -> Tensor:
        return NucleiInstances.from_seg(seg, cont, self.seg_thresh, self.cont_thresh).as_tensor()


class DistPostProcess(Postprocess):
    """
    Postprocessing based on the distance map.

    The postprocessing strategy by Naylor et al. 2019 is used to identify nuclei instances.
    """

    def __init__(self, param: int, thresh: Union[int, float]):
        self.param = param
        self.thresh = thresh

    def __call__(self, pred: Dict[str, Tensor]) -> Union[Tensor, Tuple[Tensor, ...]]:
        return self.postprocess(pred["dist_map"])

    def postprocess_fn(self, dist: Tensor) -> Tensor:
        return NucleiInstances.from_dist_map(dist, self.param, self.thresh).as_tensor()


class HVPostProcess(Postprocess):
    """
   Postprocessing based on the horizontal and vertical distance map.

   The postprocessing strategy by Graham et al. 2019 is used to identify nuclei instances.
   """

    def __init__(self, seg_thresh: Union[str, float] = 0.5):
        self.seg_thresh = seg_thresh

    def __call__(self, pred: Dict[str, Tensor]) -> Union[Tensor, Tuple[Tensor, ...]]:
        seg = pred["seg_mask"]
        if torch.is_floating_point(seg):
            seg = torch.sigmoid(seg)

        return self.postprocess(seg, pred["hv_map"])

    def postprocess_fn(self, seg: Tensor, hv_map: Tensor) -> Tensor:
        return NucleiInstances.from_hv_map(hv_map, seg, self.seg_thresh).as_tensor()

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

