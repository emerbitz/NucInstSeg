from abc import ABC, abstractmethod
from typing import Tuple, Union

from torch import Tensor


class Score(ABC):
    """Abstract base class for the implementation of custom evaluation metrics with torchmetrics."""

    def update(self, preds: Union[Tensor, Tuple[Tensor, ...]], target: Union[Tensor, Tuple[Tensor, ...]]) -> None:
        """
        Debatches and feeds the predicted nuclei instances and target nuclei instances into the evaluation function.
        """

        if isinstance(preds, tuple):  # Batch size > 1
            if len(preds) != len(target):
                raise ValueError(f"Preds should match the number of items in target. Got instead {len(preds)} and "
                                 f"{len(target)} items for preds and target, respectively.")
            for pred_inst, gt_inst in zip(preds, target):
                self.evaluate(pred_inst=pred_inst, gt_inst=gt_inst)
        elif isinstance(preds, Tensor):  # Batch size = 1
            self.evaluate(pred_inst=preds.squeeze(), gt_inst=target.squeeze())
        else:
            raise TypeError(f"Preds should be of type tuple or tensor. Got instead {type(preds)}.")

    @abstractmethod
    def evaluate(self, pred_inst: Tensor, gt_inst: Tensor) -> None:
        """
        Parameters:
        pred_inst: Tensor of shape (C, H, W)
        gt_inst: Tensor of shape (C, H, W)
        """
        pass
