from typing import Optional, Dict, Union, Any
import torch
from torch import Tensor
from torchmetrics import Metric

from evaluation.utils import tensor_intersection, intersection_over_union, tensor_union, is_empty
from evaluation.metrics_base import Score


class PQ(Score, Metric):
    """Implementation of the Panoptic Quality (PQ) using torchmetrics. PQ is the product of the Detection Quality (DQ)
    (i.e., the F1-Score) and the Segmentation Quality (SQ). See Kirillov et al. 2019 for more details."""

    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        self.add_state("TP", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum", persistent=False)
        self.add_state("FN", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum", persistent=False)
        self.add_state("FP", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum", persistent=False)
        self.add_state("IoU", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum", persistent=False)

    def evaluate(self, pred_inst: Tensor, gt_inst: Tensor) -> None:
        """Updates the states TP, FN, FP and IoU."""
        TP = torch.tensor(0, dtype=torch.int)
        for inst in gt_inst:
            for pred in pred_inst:
                iou = intersection_over_union(pred, inst)
                if iou > 0.5:  # IoU > 0.5 implies a unique match
                    self.IoU += iou
                    TP += 1
        self.TP += TP  # Detected nuclei
        self.FN += len(gt_inst) - TP  # Undetected nuclei
        self.FP += len(pred_inst) - TP  # Detected non-existing nuclei

    def compute(self) -> Dict[str, Tensor]:
        """Computes the Detection Quality (DQ), the Segmentation Quality (SQ) and the Panoptic Quality (PQ)."""
        dq = 2 * self.TP / (2 * self.TP + self.FN + self.FP)
        sq = self.IoU / self.TP
        return {"DQ": dq, "SQ": sq, "PQ": dq * sq}


# class F1Score(Score, Metric):
#     is_differentiable: Optional[bool] = False
#     higher_is_better: Optional[bool] = True
#     full_state_update: bool = False
#
#     def __init__(self):
#         super().__init__()
#         self.add_state("TP", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum", persistent=False)
#         self.add_state("FN", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum", persistent=False)
#         self.add_state("FP", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum", persistent=False)
#
#     def evaluate(self, pred_inst: torch.Tensor, gt_inst: torch.Tensor) -> NoReturn:
#         TP = torch.tensor(0, dtype=torch.int)
#         for inst in gt_inst:
#             for pred in pred_inst:
#                 if intersection_over_union(pred, inst) > 0.5:  # IoU > 0.5 implies a unique match
#                     TP += 1
#         self.TP += TP  # Detected nuclei
#         self.FN += len(gt_inst) - TP  # Undetected nuclei
#         self.FP += len(pred_inst) - TP  # Detected non-existing nuclei
#
#     def compute(self) -> torch.Tensor:
#         return 2 * self.TP / (2 * self.TP + self.FN + self.FP)


# class SQ(Score, Metric):
#     is_differentiable: Optional[bool] = False
#     higher_is_better: Optional[bool] = True
#     full_state_update: bool = False
#
#     def __init__(self):
#         super().__init__()
#         self.add_state("TP", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum", persistent=False)
#         self.add_state("IoU", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum", persistent=False)
#
#     def evaluate(self, pred_inst: torch.Tensor, gt_inst: torch.Tensor) -> NoReturn:
#         for inst in gt_inst:
#             for pred in pred_inst:
#                 iou = intersection_over_union(pred, inst)
#                 if iou > 0.5:  # IoU > 0.5 implies a unique match
#                     self.IoU += iou
#                     self.TP += 1
#
#     def compute(self) -> torch.Tensor:
#         return self.IoU / self.TP


class AJI(Score, Metric):
    """Implementation of the Aggregated Jaccard Index (AJI) using torchmetrics.
    See Kumar et al. 2017 for more details."""
    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        self.add_state("intersection", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum", persistent=False)
        self.add_state("union", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum", persistent=False)
        self.add_state("false_positives", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum",
                       persistent=False)

    def evaluate(self, pred_inst: Tensor, gt_inst: Tensor) -> None:
        """Updates the states intersection, union and false_positives."""
        if not is_empty(pred_inst):
            used_index = []
            for inst in gt_inst:
                iou = []
                for pred in pred_inst:
                    iou.append(intersection_over_union(pred, inst))
                max_iou = max(iou)
                j = iou.index(max_iou)
                self.intersection += tensor_intersection(pred_inst[j], inst)
                self.union += tensor_union(pred_inst[j], inst)
                used_index.append(j)
            for index in range(len(pred_inst)):
                if index not in used_index:
                    self.false_positives += pred_inst[index].sum()
        else:
            self.union += gt_inst.sum()

    def compute(self) -> Dict[str, Tensor]:
        """Computes the Aggregated Jaccard Index (AJI)."""
        return {"AJI": self.intersection / (self.union + self.false_positives)}


# class Proposed(Score, Metric):
#     is_differentiable: Optional[bool] = False
#     higher_is_better: Optional[bool] = True
#     full_state_update: bool = False
#
#     def __init__(self):
#         super().__init__()
#         self.add_state("intersection", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum", persistent=False)
#         self.add_state("ground_truth", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum", persistent=False)
#         self.add_state("predicted", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum", persistent=False)
#
#     def evaluate(self, pred_inst: torch.Tensor, gt_inst: torch.Tensor) -> NoReturn:
#         for inst in gt_inst:
#             intersection_list = []
#             for pred in pred_inst:
#                 intersection_list.append(tensor_intersection(pred, inst))
#             self.intersection += max(intersection_list)
#         self.ground_truth += gt_inst.sum()
#         self.predicted += pred_inst.sum()
#
#     def compute(self) -> torch.Tensor:
#         return self.intersection / (self.ground_truth + self.predicted - self.intersection)

if __name__ == "__main__":
    from tqdm import tqdm

    from data.MoNuSeg.dataset import MoNuSeg
    from data.MoNuSeg.illustrator import Picture
    from data.MoNuSeg.data_module import MoNuSegDataModule
    from postprocessing.segmentation import InstanceExtractor

    data_module = MoNuSegDataModule(
        seg_masks=True,
        cont_masks=True,
        dist_maps=False,
        labels=True,
        data_root="datasets"
    )

    # data_module.prepare_data()
    data_module.setup(stage="test")
    test_loader = data_module.test_dataloader()

    pq_metric = PQ()
    aji_metric = AJI()

    for batch, (imgs, seg_maps, cont_maps, instances, labels) in enumerate(test_loader):
        print(labels)
        pred = InstanceExtractor(seg=seg_maps, cont=cont_maps).get_instances()

        pq_metric.update(pred, instances)

        aji_metric.update(pred, instances)
        if batch == 3:
            break

    f1, sq, pq = pq_metric.compute().values
    print(f"F1-Score: {f1}")
    print(f"Segmentation Quality (SQ): {sq}")
    print(f"Panoptic Quality (PQ): {pq}")
    aji = aji_metric.compute()
    print(f"Aggregated Jaccard Index (AJI): {aji}")