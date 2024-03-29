import functools
import warnings
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torchmetrics import MetricCollection

from data.MoNuSeg.illustrator import Picture
from evaluation.metrics import DSC, ModAJI, PQ
from evaluation.utils import is_empty
from models.losses import SegLoss, DistLoss, HVLoss
from postprocessing.postprocesses import SegPostProcess, DistPostProcess, HVPostProcess


class NetModule(pl.LightningModule):
    """
    General module for training, validating, and testing of neural networks.

    The network module contains
    * The training, validation, and test loop
    * The logic for the optimizer and learning rate schedulers
    * Auxiliary functions for the logging of network predictions.
    """

    def __init__(self, net, train_params: Optional[Dict] = None, pprocess_params: Optional[Dict] = None,
                 enable_val_metrics: bool = False) -> None:
        super().__init__()
        # # Use in combination with the detailed ModelSummary callback:
        # self.example_input_array = Tensor(8, 3, 256, 256)
        self.save_hyperparameters(ignore=["net"])
        self.net = net
        self.enable_val_metrics = enable_val_metrics

        default_train_params = {
            "lr": 6e-4,
            "min_lr": 0,  # 1e-7,
            "max_epochs": 200,
            "warmup_epochs": 3,
            "period_initial": 50,
            "period_mult": 1,
            "reduction_factor": 0.5,
            "patience": 10,
            "batch_size": 8,
            "lr_schedule": "warmup_cos",
            "start_pprocess_epoch": 5
        }
        if train_params is None:
            train_params = {}
        self.train_params = {**default_train_params, **train_params}
        self.pprocess_params = pprocess_params

        self.mode = getattr(net, "mode")
        double_main_task = getattr(net, "double_main_task", False)
        self.auxiliary_task = getattr(net, "auxiliary_task", True)

        if self.mode in ["baseline", "contour", "yang"]:
            self.loss_fn = SegLoss(double_main_task=double_main_task, auxiliary_task=self.auxiliary_task)
            self.postprocess_fn = SegPostProcess(seg_params=self.pprocess_params, mode=self.mode)
        elif self.mode == "naylor":
            naylor_aux_task = getattr(net, "naylor_aux_task", "cont_mask")
            self.loss_fn = DistLoss(double_main_task=double_main_task, auxiliary_task=self.auxiliary_task,
                                    aux_task_mode=naylor_aux_task)
            self.postprocess_fn = DistPostProcess(dist_params=self.pprocess_params)
        elif self.mode in ["exprmtl", "graham"]:
            self.loss_fn = HVLoss(double_main_task=double_main_task)
            # self.loss_fn = GrahamLoss(additional_loss_term=additional_loss_term)
            self.postprocess_fn = HVPostProcess(hv_params=self.pprocess_params, exprmtl=self.mode == "exprmtl")
        else:
            raise ValueError(f"Mode should be 'baseline', 'contour', 'yang', 'naylor', 'graham' or 'exprmtl'. "
                             f"Got instead {self.mode}.")

        metrics = MetricCollection([
            DSC(),
            # AJI(),
            ModAJI(),
            PQ()
        ], compute_groups=False)
        self.val_metrics = metrics.clone()
        self.test_metrics = metrics.clone()

        self.log_on_epoch = functools.partial(self.log, on_step=False, on_epoch=True,
                                              batch_size=self.train_params["batch_size"])
        self.log_dict_on_epoch = functools.partial(self.log_dict, on_step=False, on_epoch=True,
                                                   batch_size=self.train_params["batch_size"])

    def forward(self, x: Tensor) -> Dict[str, Union[None, Tensor]]:
        return self.net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.train_params["lr"])
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.train_params["lr"])

        if self.train_params["lr_schedule"] == "none":
            return {"optimizer": optimizer}
        elif self.train_params["lr_schedule"] == "warmup_cos":
            scheduler = SequentialLR(optimizer, milestones=[self.train_params["warmup_epochs"]], schedulers=[
                LinearLR(optimizer, total_iters=self.train_params["warmup_epochs"]),
                CosineAnnealingLR(optimizer, eta_min=self.train_params["min_lr"],
                                  T_max=self.train_params["max_epochs"] - self.train_params["warmup_epochs"])
            ])
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.train_params["lr_schedule"] == "cos_warm_restart":
            scheduler = CosineAnnealingWarmRestarts(optimizer, eta_min=self.train_params["min_lr"],
                                                    T_0=self.train_params["period_initial"],
                                                    T_mult=self.train_params["period_mult"])
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.train_params["lr_schedule"] == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                   factor=self.train_params["reduction_factor"],
                                                                   patience=self.train_params["patience"],
                                                                   min_lr=self.train_params["min_lr"])
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        else:
            raise ValueError(f"Lr schedule should be either 'none', 'plateau', 'warmup_cos' or "
                             f"'cos_warm_restart'. Got instead {self.train_params['lr_schedule']}.")

    def training_step(self, train_batch):
        pred = self.forward(train_batch["img"])
        loss = self.loss_fn(pred, train_batch)

        self.log_on_epoch("lr", self.optimizers().param_groups[-1]['lr'])
        self.log_on_epoch("step", torch.tensor(self.current_epoch, dtype=torch.float))
        self.log_on_epoch("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        pred = self.forward(val_batch["img"])
        loss = self.loss_fn(pred, val_batch)

        # No postprocessing for the first epochs
        if self.current_epoch >= self.train_params["start_pprocess_epoch"]:
            if self.enable_val_metrics:
                pred["inst"] = self.postprocess_fn(pred)
                self.val_metrics.update(pred["inst"], val_batch["inst"])

        self.log_on_epoch("step", torch.tensor(self.current_epoch, dtype=torch.float))
        self.log_on_epoch("val_loss", loss)

        return loss

    def on_validation_epoch_end(self) -> None:
        # Catch UserWarning resulting from calling the 'compute' before the 'update' method
        if self.enable_val_metrics:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self.log_dict(self.val_metrics.compute())
            self.val_metrics.reset()

    def test_step(self, test_batch, batch_idx):
        pred = self.forward(test_batch["img"])
        loss = self.loss_fn(pred, test_batch)
        pred["inst"] = self.postprocess_fn(pred)

        self.log_on_epoch("step", torch.tensor(self.current_epoch, dtype=torch.float))
        self.log_on_epoch("test_loss", loss)

        self.test_metrics.update(pred["inst"], test_batch["inst"])

        if batch_idx == 0:
            self.log_pred_and_gt(pred, test_batch, idx=4)

        if "label" in test_batch.keys():
            self.log_pred_and_gt_for_labels(pred, test_batch, labels=["TCGA-AY-A8YK-01A-01-TS1_256_3",
                                                                      "TCGA-B0-5698-01Z-00-DX1_256_1",
                                                                      "TCGA-21-5784-01Z-00-DX1_256_0"])

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    @property
    def device(self) -> str:
        """Returns the device of the model."""
        return self._device

    def log_pred_and_gt_for_labels(self, pred: Dict[str, Union[Tensor, Tuple[Tensor, ...]]],
                                   gt: Dict[str, Union[Tensor, Tuple[Tensor, ...]]], labels: List[str]):
        """Logs the network predictions and associated ground truth representations for the given labels."""
        for i, l in enumerate(gt["label"]):
            if l in labels:
                self.log_pred_and_gt(pred, gt, idx=i, name=l)

    def log_pred_and_gt(self, pred: Dict[str, Union[Tensor, Tuple[Tensor, ...]]],
                        gt: Dict[str, Union[Tensor, Tuple[Tensor, ...]]], idx: int = 0, name: Optional[str] = None):
        """Logs the network predictions and associated ground truth representations."""
        self.log_pred(pred, idx=idx, name=name)
        self.log_gt(gt, idx=idx, name=name)

    def log_gt(self, gt: Dict[str, Union[Tensor, Tuple[Tensor, ...]]], idx: int = 0, name: Optional[str] = None):
        """Logs the ground truth representations."""
        for gt_type, tensor in gt.items():
            if gt_type in ["img", "seg_mask", "cont_mask"]:
                tag = gt_type if gt_type == "img" else f"{gt_type}_gt"
                if name is not None:
                    tag = f"{tag}_{name}"
                self.log_image(tensor, tag=tag, idx=idx)
            elif gt_type == "dist_map":
                tag = "dist_map_gt"
                if name is not None:
                    tag = f"{tag}_{name}"
                self.log_map(tensor, tag=[tag], idx=idx)
            elif gt_type == "hv_map":
                if name is not None:
                    self.log_map(tensor, tag=[f"h_map_gt_{name}", f"v_map_gt_{name}"], idx=idx)
                else:
                    self.log_map(tensor, tag=["h_map_gt", "v_map_gt"], idx=idx)
            elif gt_type == "inst":
                tag = "inst_gt"
                if name is not None:
                    tag = f"{tag}_{name}"
                self.log_instances(tensor, tag=tag, idx=idx)

    def log_pred(self, pred: Dict[str, Union[Tensor, Tuple[Tensor, ...]]], idx: int = 0, name: Optional[str] = None):
        """Logs the network predictions."""
        for pred_type, tensor in pred.items():
            if pred_type in ["seg_mask", "cont_mask"]:
                thresh_key, thresh_default = ("thresh_seg", 0.5) if pred_type == "seg_mask" else ("thresh_cont", 0.15)
                thresh = self.pprocess_params.get(thresh_key, thresh_default)
                tag = f"{pred_type}_pred"
                if name is not None:
                    tag = f"{tag}_{name}"
                self.log_mask(tensor, tag=tag, thresh=thresh, idx=idx)
            elif pred_type == "dist_map":
                tag = "dist_map_pred"
                if name is not None:
                    tag = f"{tag}_{name}"
                self.log_map(tensor, tag=[tag], idx=idx)
            elif pred_type == "hv_map":
                if name is not None:
                    self.log_map(tensor, tag=[f"h_map_pred_{name}", f"v_map_pred_{name}"], idx=idx)
                else:
                    self.log_map(tensor, tag=["h_map_pred", "v_map_pred"], idx=idx)
            elif pred_type == "inst":
                tag = "inst_pred"
                if name is not None:
                    tag = f"{tag}_{name}"
                self.log_instances(tensor, tag=tag, idx=idx)

    def log_instances(self, inst: Tuple[Tensor, ...], img: Optional[Tensor] = None, tag: str = "inst",
                      idx: int = 0):
        """
        Logs a color-coded image of the nuclei instances.

        If an image is additionally given, the logged image is an overlay of the color-coded instances and the provided
        image.
        """
        inst = inst[idx]
        if img is not None:
            img = img[idx]
        if not is_empty(inst):
            img = Picture.create_colored_inst(inst, img)
            self._log_img(img, tag=tag)
        else:
            img = np.zeros((256, 256), dtype=bool)  # Img size hard coded
            self._log_img(img, tag=tag)

    def log_mask(self, img: Tensor, thresh: Union[str, float] = 0.5, tag: str = "mask", idx: int = 0):
        """Logs the thresholded image."""
        img = torch.sigmoid(img)
        img = img[idx]
        img = Picture.create_mask(img, thresh)
        self._log_img(img, tag=tag)

    def log_map(self, img: Tensor, tag: List[str], idx: int = 0):
        """Logs the distance maps."""
        for i, t in zip(img[idx], tag):
            i = Picture.tensor_to_ndarray(i)
            if t[:8] == "dist_map":
                i = cv2.normalize(i, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            if t[:5] == "h_map" or t[:5] == "v_map":
                i = cv2.normalize(i, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                cmap = plt.get_cmap("bwr")
                i = cmap(i)
            self._log_img(img=i, tag=t)

    def log_image(self, img: Tensor, tag: str = "image", idx: int = 0):
        img = img[idx]
        img = Picture.tensor_to_ndarray(img)
        self._log_img(img, tag=tag)

    def _log_img(self, img: np.ndarray, tag: str = "image"):
        tensorboard = self.logger.experiment
        data_format = "HWC" if img.ndim == 3 else "HW"
        tensorboard.add_image(tag=tag, img_tensor=img, dataformats=data_format)
