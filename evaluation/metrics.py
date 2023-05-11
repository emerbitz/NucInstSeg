from typing import Optional, Dict, Union, Any
from scipy.optimize import linear_sum_assignment
import numpy as np
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

    def evaluate(self, pred_inst: Tensor, gt_inst: Tensor) -> None:
        """Updates the states intersection and union."""
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
                    self.union += pred_inst[index].sum()
        else:
            self.union += gt_inst.sum()

    def compute(self) -> Dict[str, Tensor]:
        """Computes the Aggregated Jaccard Index (AJI)."""
        return {"AJI": self.intersection / self.union}

class AJI_v1(Score, Metric):
    """
    Implementation of the Aggregated Jaccard Index (AJI) using torchmetrics.
    See Kumar et al. 2017 for more details.

    version 1
    """
    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        self.add_state("intersection", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum", persistent=False)
        self.add_state("union", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum", persistent=False)

    def evaluate(self, pred_inst: Tensor, gt_inst: Tensor) -> None:
        """Updates the states intersection and union."""
        if not is_empty(pred_inst):
            used_index = []
            num_pred = pred_inst.shape[0]
            for inst in gt_inst:
                inter = torch.zeros(num_pred, dtype=torch.int)
                union = torch.zeros(num_pred, dtype=torch.int)
                for i, pred in enumerate(pred_inst):
                    inter[i] = tensor_intersection(pred, inst)
                    union[i] = tensor_union(pred, inst)
                iou = inter / (union + 1e-6)  # Avoid zero division
                j = torch.argmax(iou)
                self.intersection += inter[j]
                self.union += union[j]
                used_index.append(j)
            # Adds the unassigned predictions to the union:
            for idx in range(num_pred):
                if idx not in used_index:
                    self.union += pred_inst[idx].sum()
        else:
            self.union += gt_inst.sum()

    def compute(self) -> Dict[str, Tensor]:
        """Computes the Aggregated Jaccard Index (AJI)."""
        return {"AJI": self.intersection / self.union}

class AJI_v2(Score, Metric):
    """
    Implementation of the Aggregated Jaccard Index (AJI) using torchmetrics.
    See Kumar et al. 2017 for more details.

    version 2
    """
    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        self.add_state("intersection", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum", persistent=False)
        self.add_state("union", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum", persistent=False)

    def evaluate(self, pred_inst: Tensor, gt_inst: Tensor) -> None:
        """Updates the states intersection and union."""
        if not is_empty(pred_inst):
            num_pred = pred_inst.shape[0]
            num_gt = gt_inst.shape[0]
            pairwise_inter = torch.zeros((num_pred, num_gt), dtype=torch.int)
            pairwise_union = torch.zeros((num_pred, num_gt), dtype=torch.int)
            for col, inst in enumerate(gt_inst):
                for row, pred in enumerate(pred_inst):
                    pairwise_inter[row, col] = tensor_intersection(pred, inst)
                    pairwise_union[row, col] = tensor_union(pred, inst)

            iou = pairwise_inter / (pairwise_union + 1e-6)  # Avoid zero division
            row_idx = torch.argmax(iou, dim=0)
            col_idx = torch.arange(0, num_gt, dtype=torch.long)
            self.intersection += pairwise_inter[row_idx, col_idx].sum()
            self.union += pairwise_union[row_idx, col_idx].sum()
            # Add the unassigned predictions and ground truths to the union:
            self.union += sum([pred_inst[r].sum() for r in range(num_pred) if r not in row_idx])
            self.union += sum([gt_inst[c].sum() for c in range(num_gt) if c not in col_idx])
        else:
            self.union += gt_inst.sum()

    def compute(self) -> Dict[str, Tensor]:
        """Computes the Aggregated Jaccard Index (AJI)."""
        return {"AJI": self.intersection / self.union}


class ModAJI(Score, Metric):
    """
    Implementation of a modified Aggregated Jaccard Index (AJI) using torchmetrics.

    The AJI (Kumar et al. 2017) calculates the argmax(Intersection over Union (IoU)) to match predicted instances with
    ground truth instances. However, unambiguous matches are only ensured for IoU values greater than 0.5 (see Kirillov
    et al. 2019 for proof). A predicted instance might be assigned to multiple ground truth instances for IoU values
    smaller or equal 0.5. This one-to-many matching may lead to over-penalization (Graham et al. 2019).

    The modified AJI calculates the Hungarian algorithm for the IoU values to match predicted instances with ground
    truth instances. The Hungarian algorithm ensures one-to-one matching: A predicted instance is assigned at most to a
    single ground truth instance. Thus, the modified AJI is greater or equal to the AJI, as over-penalization is
    prevented.
    """
    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        self.add_state("intersection", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum", persistent=False)
        self.add_state("union", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum", persistent=False)

    def evaluate(self, pred_inst: Tensor, gt_inst: Tensor) -> None:
        """Updates the states intersection and union."""
        if not is_empty(pred_inst):
            num_pred = pred_inst.shape[0]
            num_gt = gt_inst.shape[0]
            pairwise_inter = torch.zeros((num_pred, num_gt), dtype=torch.int)
            pairwise_union = torch.zeros((num_pred, num_gt), dtype=torch.int)
            for col, inst in enumerate(gt_inst):
                for row, pred in enumerate(pred_inst):
                    pairwise_inter[row, col] = tensor_intersection(pred, inst)
                    pairwise_union[row, col] = tensor_union(pred, inst)

            iou = pairwise_inter / (pairwise_union + 1e-6)  # Avoid zero division
            # Optimization using a modified Jonker-Volgenant algorithm:
            row_ind, col_ind = linear_sum_assignment(-iou)  # Negate IoU values as algo searches for minimum
            self.intersection += pairwise_inter[row_ind, col_ind].sum()
            self.union += pairwise_union[row_ind, col_ind].sum()
            # Add the unassigned predictions and ground truths to the union:
            self.union += sum([pred_inst[r].sum() for r in range(num_pred) if r not in row_ind])
            self.union += sum([gt_inst[c].sum() for c in range(num_gt) if c not in col_ind])
        else:
            self.union += gt_inst.sum()

    def compute(self) -> Dict[str, Tensor]:
        """Computes the Aggregated Jaccard Index (AJI)."""
        return {"AJI": self.intersection / self.union}

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