from typing import NoReturn
import pytorch_lightning as pl
import torch
import torch.nn as nn


class ALNetModel(pl.LightningModule):
    def __init__(self, model) -> NoReturn:
        super(ALNetModel, self).__init__()
        self.model = model
        self.lr = 2e-4
        self.lambda_main, self.lambda_aux = 1, 1
        self.loss_main = nn.BCEWithLogitsLoss()  # Penalty term to be added
        self.loss_aux = nn.BCEWithLogitsLoss()

    def forward(self, x) -> torch.Tensor:
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
        imgs, seg_masks, cont_masks = val_batch
        seg, cont = self.forward(imgs)
        loss = self.lambda_main * self.loss_main(seg, seg_masks) + self.lambda_aux * self.loss_aux(cont, cont_masks)
        self.log("val_loss", loss)

    def test_step(self, test_batch, batch_idx):
        pass