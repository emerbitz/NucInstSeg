import pytorch_lightning as pl
import torch
from torch import Tensor
import torch.nn as nn

from data.MoNuSeg.illustrator import Picture
from postprocessing.segmentation import InstanceExtractor
from evaluation.metrics import PQ, AJI

from transformation.transformations import PadZeros
from augmentation.augmentations import RandCrop


class ALNetModel(pl.LightningModule):
    # def __init__(self, model, lr: int = 2e-4, lambda_main: int = 1, lambda_aux: int = 1) -> None:
    def __init__(self, model) -> None:
        super(ALNetModel, self).__init__()
        # Use in combination with the detailed ModelSummary callback
        # self.example_input_array = Tensor(8, 3, 256, 256)
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.lr = lr = 2e-4  # 2e-4
        self.lambda_main, self.lambda_aux = lambda_main, lambda_aux = 1, 1
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

        # self.log("train_loss_per_step", loss, on_step=True, on_epoch=False)
        self.log("step", torch.tensor(self.current_epoch, dtype=torch.float), on_step=False, on_epoch=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        imgs, seg_masks, cont_masks, instances = val_batch
        seg, cont = self.forward(imgs)
        loss = self.lambda_main * self.loss_main(seg, seg_masks) + self.lambda_aux * self.loss_aux(cont, cont_masks)

        seg_log, cont_log = torch.sigmoid(seg), torch.sigmoid(cont)
        pred_instances = InstanceExtractor(seg=seg_log, cont=cont_log).get_instances(impl="skimage")
        if batch_idx == 0 and self.current_epoch % 2 == 0:
            Picture.from_tensor((seg_log[0] > 0.5).cpu()).save("seg_log_" + str(self.current_epoch))
            # self.log("seg_log" + str(self.current_epoch), seg_log[0, 0] > 0.5)
            Picture.from_tensor((cont_log[0] > 0.5).cpu()).save("cont_log_" + str(self.current_epoch))
            # self.log("cont_log" + str(self.current_epoch), cont_log[0, 0] > 0.5)
        if batch_idx == 0 and self.current_epoch == 0:  # Change to zero
            Picture.from_tensor(seg_masks[0].cpu()).save("seg_mask")
            Picture.from_tensor(cont_masks[0].cpu()).save("cont_mask")

        self.pq_val.update(pred_instances, instances)
        self.aji_val.update(pred_instances, instances)

        # self.log("val_loss_per_step", loss, on_step=True, on_epoch=False)
        self.log("step", torch.tensor(self.current_epoch, dtype=torch.float), on_step=False, on_epoch=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log_dict(self.pq_val.compute(), on_step=False, on_epoch=True)
        self.log_dict(self.aji_val.compute(), on_step=False, on_epoch=True)
        return loss

    # def validation_epoch_end(self, outputs):
    #     self.log("step", torch.tensor(self.current_epoch, dtype=torch.float))

    def test_step(self, test_batch, batch_idx):
        imgs, seg_masks, cont_masks, instances = test_batch
        seg, cont = self.forward(imgs)
        loss = self.lambda_main * self.loss_main(seg, seg_masks) + self.lambda_aux * self.loss_aux(cont, cont_masks)
        self.log("test_loss", loss)

        pred_instances = InstanceExtractor(seg=seg, cont=cont).get_instances()
        self.pq_test.update(pred_instances, instances)
        self.log_dict(self.pq_test.compute(), on_step=False, on_epoch=True)
        self.aji_test.update(pred_instances, instances)
        self.log_dict(self.aji_test.compute(), on_step=False, on_epoch=True)

