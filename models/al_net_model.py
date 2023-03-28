import pytorch_lightning as pl
import torch
from torch import Tensor
import torch.nn as nn

from postprocessing.segmentation import NucleiSplitter
from evaluation.metrics import PQ, AJI


class ALNetModel(pl.LightningModule):
    def __init__(self, model) -> None:
        super(ALNetModel, self).__init__()
        self.model = model
        self.lr = 2e-4
        self.lambda_main, self.lambda_aux = 1, 1
        self.loss_main = nn.BCEWithLogitsLoss()  # Penalty term to be added
        self.loss_aux = nn.BCEWithLogitsLoss()

        self.pq_val, self.pq_test = PQ(), PQ()
        self.aji_val, self.aji_test = AJI(), AJI()

    def forward(self, x) -> Tensor:
        return self.model(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, train_batch, batch_idx):
        imgs, seg_masks, cont_masks = train_batch
        seg, cont = self.forward(imgs)
        loss = self.lambda_main * self.loss_main(seg, seg_masks) + self.lambda_aux * self.loss_aux(cont, cont_masks)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        imgs, seg_masks, cont_masks, instances = val_batch
        seg, cont = self.forward(imgs)
        loss = self.lambda_main * self.loss_main(seg, seg_masks) + self.lambda_aux * self.loss_aux(cont, cont_masks)
        self.log("val_loss", loss)

        pred_instances = NucleiSplitter(seg=seg, cont=cont).to_instances()
        self.pq_val.update(pred_instances, instances)
        self.aji_val.update(pred_instances, instances)

    def validation_epoch_end(self, outputs):
        dq, sq, pq = self.pq_val.compute()
        self.log("val_dq", dq)
        self.log("val_sq", sq)
        self.log("val_pq", pq)
        self.log("val_aji", self.aji_val.compute())
        self.pq_val.reset(), self.aji_val.reset()

    def test_step(self, test_batch, batch_idx):
        imgs, seg_masks, cont_masks, instances = test_batch
        seg, cont = self.forward(imgs)
        loss = self.lambda_main * self.loss_main(seg, seg_masks) + self.lambda_aux * self.loss_aux(cont, cont_masks)
        self.log("test_loss", loss)

        pred_instances = NucleiSplitter(seg=seg, cont=cont).to_instances()
        self.pq_test.update(pred_instances, instances)
        self.aji_test.update(pred_instances, instances)

    def test_epoch_end(self, outputs):
        dq, sq, pq = self.pq_test.compute()
        self.log("test_dq", dq)
        self.log("test_sq", sq)
        self.log("test_pq", pq)
        self.log("test_aji", self.aji_test.compute())
        self.pq_test.reset(), self.aji_test.reset()