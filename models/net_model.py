import functools
from typing import Dict, List, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor

from data.MoNuSeg.illustrator import Picture
from evaluation.metrics import DSC, AJI, ModAJI, PQ
from evaluation.utils import is_empty
from models.losses import SegLoss, DistLoss, HVLoss
from postprocessing.postprocesses import SegPostProcess, DistPostProcess, HVPostProcess


class NetModel(pl.LightningModule):

    def __init__(self, net, lr: float = 1e-4, batch_size: int = 8) -> None:
        super().__init__()
        # # Use in combination with the detailed ModelSummary callback:
        # self.example_input_array = Tensor(8, 3, 256, 256)
        self.save_hyperparameters(ignore=["net", "batch_size"])
        self.net = net

        self.mode = net.mode
        if self.mode == "seg":
            self.loss_fn = SegLoss()
            self.postprocess_fn = SegPostProcess(seg_thresh=0.5, cont_thresh=0.5)
        elif self.mode == "dist":
            self.loss_fn = DistLoss()
            self.postprocess_fn = DistPostProcess(param=3, thresh=0)
        elif self.mode == "hv":
            self.loss_fn = HVLoss()
            self.postprocess_fn = HVPostProcess(seg_thresh=0.5)
        else:
            raise ValueError(f"Mode should be seg, dist or hv. Got instead {self.mode}.")

        self.lr = lr
        self.min_lr = 0.0000001
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.val_metrics = [
            DSC().to(self._device),
            AJI().to(self._device),
            ModAJI().to(self._device),
            PQ().to(self._device)
        ]
        self.test_metrics = [
            DSC().to(self._device),
            AJI().to(self._device),
            ModAJI().to(self._device),
            PQ().to(self._device)
        ]

        self.log_on_epoch = functools.partial(self.log, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log_dict_on_epoch = functools.partial(self.log_dict, on_step=False, on_epoch=True, batch_size=batch_size)

    def forward(self, x: Tensor) -> Dict[str, Union[None, Tensor]]:
        return self.net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4, min_lr=self.min_lr)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=self.min_lr)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=self.min_lr)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, train_batch, batch_idx):

        pred = self.forward(train_batch["img"])
        loss = self.loss_fn(pred, train_batch)

        # Picture.from_tensor(train_batch["hv_map"][0][0]).save(file_name=f"gt_h_map_{self.current_epoch}")
        # Picture.from_tensor(train_batch["hv_map"][0][1]).save(file_name=f"gt_v_map_{self.current_epoch}")
        # Picture.from_tensor(pred["hv_map"][0][0]).save(file_name=f"pred_h_map_{self.current_epoch}")
        # Picture.from_tensor(pred["hv_map"][0][1]).save(file_name=f"pred_v_map_{self.current_epoch}")

        # self.log_pred(train_batch["img"], pred)
        # self.log_overlay(train_batch["img"], train_batch["inst"], tag="gt_overlay")
        # pred["inst"] = self.postprocess_fn(pred)
        # self.log_overlay(train_batch["img"], pred["inst"], tag="pred_overlay")
        # self.log_map(train_batch["hv_map"], tag=["gt_h_map", "gt_v_map"])

        self.log_on_epoch("step", torch.tensor(self.current_epoch, dtype=torch.float))
        self.log_on_epoch("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        pred = self.forward(val_batch["img"])
        loss = self.loss_fn(pred, val_batch)
        pred["inst"] = self.postprocess_fn(pred)

        self.log_on_epoch("step", torch.tensor(self.current_epoch, dtype=torch.float))
        self.log_on_epoch("val_loss", loss)
        # self.log_overlay(val_batch["img"], pred["inst"])

        for m in self.val_metrics:
            m.update(pred["inst"], val_batch["inst"])
            self.log_dict_on_epoch(m.compute())
        return loss

    def test_step(self, test_batch, batch_idx):
        pred = self.forward(test_batch["img"])
        loss = self.loss_fn(pred, test_batch)
        pred["inst"] = self.postprocess_fn(pred)

        self.log_on_epoch("step", torch.tensor(self.current_epoch, dtype=torch.float))
        self.log_on_epoch("test_loss", loss)

        for m in self.test_metrics:
            m.update(pred["inst"], test_batch["inst"])
            self.log_dict_on_epoch(m.compute())

    @property
    def device(self) -> str:
        """Returns the device of the model."""
        return self._device

    def log_pred(self, img: Tensor, pred: Dict[str, Union[Tensor, Tuple[Tensor, ...]]]):
        if self.mode == "seg":
            self.log_mask(pred["seg_mask"], thresh=0.5, tag="seg_mask")
            self.log_mask(pred["cont_mask"], thresh=0.5, tag="cont_mask")
            # self.log_overlay(img, pred["inst"])
        elif self.mode == "dist":
            self.log_map(pred["dist_map"], tag=["dist_map"])
            # self.log_overlay(img, pred["inst"])
        elif self.mode == "hv":
            self.log_mask(pred["seg_mask"], thresh=0.5, tag="seg_mask")
            self.log_map(pred["hv_map"], tag=["h_map", "v_map"])
            # self.log_overlay(img, pred["inst"])

    def log_overlay(self, img: Tensor, inst: Tuple[Tensor, ...], tag: str = "overlay", idx: int = 0):
        """Logs an overlay of the image and the nuclei instances."""
        img = img[idx]
        inst = inst[idx]
        if not is_empty(inst):
            img = Picture.create_overlay(img, inst)
        self._log_img(img, tag=tag)

    def log_mask(self, img: Tensor, thresh: Union[str, float] = 0.5, tag: str = "mask", idx: int = 0):
        """Logs the thresholded image."""
        img = img[idx]
        img = Picture.create_mask(img, thresh)
        self._log_img(img, tag=tag, order="HW")

    def log_map(self, img: Tensor, tag: List[str], idx: int = 0):
        for i, t in zip(img[idx], tag):
            i = Picture.tensor_to_ndarray(i)
            self._log_img(img=i, tag=t, order="HW")

    def _log_img(self, img: np.ndarray, tag: str = "image", order: str = "HWC"):
        tensorboard = self.logger.experiment
        tensorboard.add_image(tag=f"{tag}_{self.current_epoch}", img_tensor=img, dataformats=order)
