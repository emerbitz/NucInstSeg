from pathlib import Path
from typing import Dict, Optional

import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import SimpleProfiler  # Alternative: AdvancedProfiler(), XLAProfiler()

from data.MoNuSeg.data_module import MoNuSegDataModule
from models.net_module import NetModule
from models.u_net import UNet, UNetDualDecoder
from models.reu_net import REUNet
from models.al_net import ALNet


def select_ckpt_path(project: str, name: str, base_dir: str = "trained_models") -> str:
    """
    Selects a checkpoint path that does not already exist.
    """
    ckpt_path = Path(base_dir, project, name)
    vol = 0
    while ckpt_path.exists():
        ckpt_path = ckpt_path.with_name(f"{name}_{vol}")
        vol += 1
    return str(ckpt_path)


def train(net: nn.Module, project: str, name: str, train_params: Optional[Dict] = None,
          pprocess_params: Optional[Dict] = None):
    """
    Trains the neural network.

    :param train_params: Dict with hyperparameters (e.g., batch size, max epochs, lr schedule) for network training.
    :param pprocess_params: Dict with hyperparameters (e.g., thresholds for the classification probability maps) for
                            the postprocessing pipelines.
    """
    model = NetModule(net=net, train_params=train_params, pprocess_params=pprocess_params)
    ckpt_path = select_ckpt_path(project, name)
    print(f"Model checkpoint: {ckpt_path}")
    callbacks = [
        ModelCheckpoint(dirpath=ckpt_path, filename="top_val_{epoch}",
                        save_top_k=1, monitor="val_loss", mode="min"),
        # ModelCheckpoint(dirpath=ckpt_path, filename="top_pq_{epoch}",
        #                 save_top_k=1, monitor="PQ", mode="max"),
        # ModelCheckpoint(dirpath=f"trained_models/{project}/{name}", save_last=True,
        #                 save_top_k=0)
        # ModelCheckpoint(dirpath=f"trained_models/{project}/{name}", filename="top_basic_performance_{epoch}",
        #                 save_top_k=1, monitor="basic_performance", mode="min"),
    ]
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=model.train_params["max_epochs"], max_time="0:4:0:0",
                         auto_lr_find=False, auto_scale_batch_size=False, deterministic=False, fast_dev_run=False,
                         log_every_n_steps=1, check_val_every_n_epoch=1, enable_model_summary=True,
                         enable_checkpointing=True, limit_train_batches=None, limit_val_batches=None,
                         limit_test_batches=None,
                         logger=TensorBoardLogger(save_dir=f"lightning_logs/{project}", name=name,
                                                  default_hp_metric=False),
                         profiler=SimpleProfiler(dirpath=f"performance_logs/{project}", filename="summary"),
                         callbacks=callbacks
                         )
    data_module = MoNuSegDataModule.default_mode(mode=model.mode, auxiliary_task=model.auxiliary_task,
                                                 batch_size=model.train_params["batch_size"])
    # trainer.tune(model, data_module)  # Automatic selection of lr and batch size
    trainer.fit(model, data_module)
    # Continue training from checkpoint:
    # trainer.fit(model, data_module, ckpt_path="trained_models/seg_dual_model/last.ckp")


if __name__ == "__main__":
    ###########################################
    #      Single decoder U-Net training
    ###########################################
    train_params = {
        "lr": 1e-4,
        "min_lr": 0,  # 1e-7,
        "max_epochs": 200,
        "warmup_epochs": 3,
        "period_initial": 50,
        "period_mult": 1,
        "reduction_factor": 0.5,
        "patience": 10,
        "batch_size": 8,
        "lr_schedule": "plateau"
    }

    pl.seed_everything(42, workers=True)  # Seed for torch, numpy and python.random
    net = UNet(mode="baseline")
    name = f"baseline_unet_lr=cpp200"
    train(net=net, project="u_net", name=name, train_params=train_params, pprocess_params=None)

    pl.seed_everything(42, workers=True)  # Seed for torch, numpy and python.random
    net = UNet(mode="contour")
    name = f"contour_unet_lr=cpp200"
    train(net=net, project="u_net", name=name, train_params=train_params, pprocess_params=None)

    pl.seed_everything(42, workers=True)  # Seed for torch, numpy and python.random
    net = UNet(mode="naylor")
    name = f"naylor_unet_lr=cpp200"
    train(net=net, project="u_net", name=name, train_params=train_params, pprocess_params=None)

    pl.seed_everything(42, workers=True)  # Seed for torch, numpy and python.random
    net = UNet(mode="graham")
    name = f"graham_unet_lr=cpp200"
    train(net=net, project="u_net", name=name, train_params=train_params, pprocess_params=None)

    ###########################################
    #      Dual decoder U-Net training
    ###########################################
    train_params = {
        "lr": 1e-4,
        "min_lr": 0,  # 1e-7,
        "max_epochs": 200,
        "warmup_epochs": 3,
        "period_initial": 50,
        "period_mult": 1,
        "reduction_factor": 0.5,
        "patience": 10,
        "batch_size": 8,
        "lr_schedule": "plateau"
    }

    pl.seed_everything(42, workers=True)  # Seed for torch, numpy and python.random
    net = UNetDualDecoder(mode="contour", con_channels=None)
    name = f"contour_unetdual_nocon_lr=cpp200"
    train(net=net, project="u_net_dual", name=name, train_params=train_params, pprocess_params=None)

    pl.seed_everything(42, workers=True)  # Seed for torch, numpy and python.random
    net = UNetDualDecoder(mode="graham", con_channels=None)
    name = f"graham_unetdual_nocon_lr=cpp200"
    train(net=net, project="u_net_dual", name=name, train_params=train_params, pprocess_params=None)

    ###########################################
    #             REU-Net training
    ###########################################
    net_params = {
        "aspp_inter_channels": 96,
        "base_channels": 48,
        "base_channels_factor": 2,
        "depth": 4,
        "norm": False
    }
    train_params = {
        "lr": 6e-4,
        "min_lr": 0,  # 1e-7,
        "max_epochs": 200,
        "warmup_epochs": 3,
        "period_initial": 50,
        "period_mult": 1,
        "reduction_factor": 0.5,
        "patience": 10,
        "batch_size": 6,
        "lr_schedule": "warmup_cos"
    }

    pl.seed_everything(42, workers=True)  # Seed for torch, numpy and python.random
    net = REUNet(mode="baseline", net_params=net_params)
    name = "baseline_reunet_netchannels48_factor2_aspp96_depth4_normf_lr=reu200"
    train(net=net, project="reu_net", name=name, train_params=train_params, pprocess_params=None)

    pl.seed_everything(42, workers=True)  # Seed for torch, numpy and python.random
    net = REUNet(mode="naylor", naylor_aux_task="cont_mask", net_params=net_params)
    name = "naylor_reunet_auxcont_netchannels48_factor2_aspp96_depth4_normf_lr=reu200"
    train(net=net, project="reu_net", name=name, train_params=train_params, pprocess_params=None)
    pl.seed_everything(42, workers=True)  # Seed for torch, numpy and python.random
    net = REUNet(mode="naylor", naylor_aux_task="hv_map", net_params=net_params)
    name = "naylor_reunet_auxhv_netchannels48_factor2_aspp96_depth4_normf_lr=reu200"
    train(net=net, project="reu_net", name=name, train_params=train_params, pprocess_params=None)

    pl.seed_everything(42, workers=True)  # Seed for torch, numpy and python.random
    net = REUNet(mode="graham", net_params=net_params)
    name = "graham_reunet_netchannels48_factor2_aspp96_depth4_normf_lr=reu200"
    train(net=net, project="reu_net", name=name, train_params=train_params, pprocess_params=None)
