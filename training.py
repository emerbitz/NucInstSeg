import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import SimpleProfiler  # Alternative: AdvancedProfiler(), XLAProfiler()

from data.MoNuSeg.data_module import MoNuSegDataModule
from models.net_model import NetModel
from models.u_net import UNet


# Maybe add optimizer argument
def train(model=NetModel(net=UNet(mode="dist"), lr=1e-4, batch_size=8),
          data_module=MoNuSegDataModule(seg_masks=False, cont_masks=False, dist_maps=True, hv_maps=False, batch_size=8),
          trainer=pl.Trainer(accelerator="gpu", devices=1, max_epochs=50, max_time="0:8:0:0", auto_lr_find=False,
                             auto_scale_batch_size=False, deterministic=False, fast_dev_run=False, log_every_n_steps=1,
                             check_val_every_n_epoch=1, enable_model_summary=True, overfit_batches=0.0,
                             limit_train_batches=None, limit_val_batches=None, limit_test_batches=None,
                             logger=TensorBoardLogger(save_dir="lightning_logs", name="dist_split_model",
                                                      default_hp_metric=False),
                             profiler=SimpleProfiler(dirpath="performance_logs", filename="summary"),
                             callbacks=[
                                 ModelCheckpoint(dirpath="trained_models/dist_split", filename="top_val_{epoch}",
                                                 save_top_k=1, monitor="val_loss"),
                                 ModelCheckpoint(dirpath="trained_models/dist_split", filename="top_pq_{epoch}",
                                                 save_top_k=1, monitor="PQ", mode="max"),
                                 # ModelCheckpoint(dirpath="trained_models/hv_dual_model", filename="finish_{epoch}", save_last=True)
                             ])
          ):
    pl.seed_everything(42, workers=True)  # Seed for torch, numpy and python.random
    # trainer.tune(model, data_module)  # Automatic selection of lr and batch size
    trainer.fit(model, data_module)
    # Continue training from checkpoint:
    # trainer.fit(model, data_module, ckpt_path="trained_models/seg_dual_model/last.ckp")


if __name__ == "__main__":
    train()
