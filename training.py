from pathlib import Path
from typing import Dict, Optional

import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import SimpleProfiler  # Alternative: AdvancedProfiler(), XLAProfiler()

from data.MoNuSeg.data_module import MoNuSegDataModule
from models.net_model import NetModel
from models.u_net import UNet


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
    model = NetModel(net=net, train_params=train_params, pprocess_params=pprocess_params)
    ckpt_path = select_ckpt_path(project, name)
    print(f"Model checkpoint: {ckpt_path}")
    callbacks = [
        ModelCheckpoint(dirpath=ckpt_path, filename="top_val_{epoch}",
                        save_top_k=1, monitor="val_loss", mode="min"),
        ModelCheckpoint(dirpath=ckpt_path, filename="top_pq_{epoch}",
                        save_top_k=1, monitor="PQ", mode="max"),
        # ModelCheckpoint(dirpath=f"trained_models/{project}/{name}", save_last=True,
        #                 save_top_k=0)
        # ModelCheckpoint(dirpath=f"trained_models/{project}/{name}", filename="top_basic_performance_{epoch}",
        #                 save_top_k=1, monitor="basic_performance", mode="min"),
    ]
    # if model.mode != "naylor":
    #     callbacks.append(
    #         ModelCheckpoint(dirpath=f"trained_models/{project}/{name}", filename="top_advanced_performance_{epoch}",
    #                         save_top_k=1, monitor="advanced_performance", mode="min"),
    #     )
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
    # train_params = {
    #     "lr": 2e-4,  # 6e-4,
    #     "min_lr": 0,  # 1e-7,
    #     "max_epochs": 200,
    #     "warmup_epochs": 3,
    #     "period_initial": 50,
    #     "period_mult": 1,
    #     "reduction_factor": 0.5,
    #     "patience": 10,
    #     "batch_size": 8,
    #     "lr_schedule": "none"
    # }
    # pprocess_params = {
    #     "seg_thresh": 0.5,
    #     "cont_tresh": "otsu",
    #     "thresh_coarse": 250,
    #     "thresh_fine": 100,
    #     "dist_thresh": 0.5,
    #     "dist_param": 3
    # }
    # pprocess_params = {}

    # net = ALNetDualDecoder(mode="baseline")
    # name = "baseline_alnetdual_lr=reu200"
    # train(net=net, project="al_net", name=name, train_params=train_params, pprocess_params=pprocess_params)
    # net = ALNetDualDecoder(mode="noname")
    # name = "noname_alnetdual_lr=reu200"
    # train(net=net, project="al_net", name=name, train_params=train_params, pprocess_params=pprocess_params)
    # net = ALNetDualDecoder(mode="yang")
    # name = "yang_alnetdual_lr=reu200"
    # train(net=net, project="al_net", name=name, train_params=train_params, pprocess_params=pprocess_params)
    # net = ALNetDualDecoder(mode="naylor")
    # name = "naylor_alnetdual_lr=reu200"
    # train(net=net, project="al_net", name=name, train_params=train_params, pprocess_params=pprocess_params)

    # net = ALNetDualDecoder(mode="graham")
    # name = "graham_alnetdual_lr=reu200"
    # train(net=net, project="al_net", name=name, train_params=train_params, pprocess_params=pprocess_params)
    # net = ALNetDualDecoder(mode="exprmtl")
    # name = "exprmtl_alnetdual_lr=reu200"
    # train(net=net, project="al_net", name=name, train_params=train_params, pprocess_params=pprocess_params)

    # net = ALNet(mode="yang")
    # name = "yang_alnet_novolup_lr=reu200"
    # train(net=net, project="al_net", name=name, train_params=train_params, pprocess_params=pprocess_params)
    # net = ALNet(mode="naylor")
    # name = "naylor_alnet_novolup_lr=reu200"
    # train(net=net, project="al_net", name=name, train_params=train_params, pprocess_params=pprocess_params)
    # net = ALNet(mode="graham")
    # name = "graham_alnet_novolup_lr=reu200"
    # train(net=net, project="al_net", name=name, train_params=train_params, pprocess_params=pprocess_params)

    # net = ALNet(mode="yang")
    # name = "yang_alnet_lr=al200"
    # train(net=net, project="al_net", name=name, train_params=train_params, pprocess_params=pprocess_params)
    # net = ALNet(mode="naylor")
    # name = "naylor_alnet_lr=al200"
    # train(net=net, project="al_net", name=name, train_params=train_params, pprocess_params=pprocess_params)
    # net = ALNet(mode="graham")
    # name = "graham_alnet_lr=al200"
    # train(net=net, project="al_net", name=name, train_params=train_params, pprocess_params=pprocess_params)
    #
    # train_params = {
    #     "lr": 1e-4,  # 6e-4,
    #     "min_lr": 1e-7,
    #     "max_epochs": 200,
    #     "warmup_epochs": 3,
    #     "period_initial": 50,
    #     "period_mult": 1,
    #     "reduction_factor": 0.5,
    #     "patience": 5,
    #     "batch_size": 8,
    #     "lr_schedule": "plateau"
    # }
    #
    # net = ALNet(mode="yang")
    # name = "yang_alnet_lr=cpp200"
    # train(net=net, project="al_net", name=name, train_params=train_params, pprocess_params=pprocess_params)
    # net = ALNet(mode="naylor")
    # name = "naylor_alnet_lr=cpp200"
    # train(net=net, project="al_net", name=name, train_params=train_params, pprocess_params=pprocess_params)
    # net = ALNet(mode="graham")
    # name = "graham_alnet_lr=cpp200"
    # train(net=net, project="al_net", name=name, train_params=train_params, pprocess_params=pprocess_params)

    train_params = {
        "lr": 6e-4,
        "min_lr": 0,  # 1e-7,
        "max_epochs": 200,
        "warmup_epochs": 3,
        "period_initial": 50,
        "period_mult": 1,
        "reduction_factor": 0.5,
        "patience": 10,
        "batch_size": 8,
        "lr_schedule": "warmup_cos"
    }

    # net_params = {
    #     "base_channels": 32,
    #     "aspp_inter_channels": 32,
    #     "base_channels_factor": 2
    # }
    # net = REUNet(mode="yang", net_params=net_params)
    # name = f"yang_reunet_channels{net_params['base_channels']}_factor{net_params['base_channels_factor']}_aspp{net_params['aspp_inter_channels']}_lr=reu200"
    # train(net=net, project="reu_net", name=name, train_params=train_params, pprocess_params=None)
    # net = REUNet(mode="naylor", net_params=net_params)
    # name = f"naylor_reunet_channels{net_params['base_channels']}_factor{net_params['base_channels_factor']}_aspp{net_params['aspp_inter_channels']}_lr=reu200"
    # train(net=net, project="reu_net", name=name, train_params=train_params, pprocess_params=None)
    # net = REUNet(mode="graham", net_params=net_params)
    # name = f"graham_reunet_channels{net_params['base_channels']}_factor{net_params['base_channels_factor']}_aspp{net_params['aspp_inter_channels']}_lr=reu200"
    # train(net=net, project="reu_net", name=name, train_params=train_params, pprocess_params=None)

    # net_params = {
    #     "aspp_inter_channels": 64,
    #     "base_channels": 64,
    #     "base_channels_factor": 2
    # }
    # net = REUNet(mode="yang", net_params=net_params)
    # name = f"yang_reunet_channels{net_params['base_channels']}_factor{net_params['base_channels_factor']}_aspp{net_params['aspp_inter_channels']}_lr=reu200"
    # train(net=net, project="reu_net", name=name, train_params=train_params, pprocess_params=None)
    # net = REUNet(mode="naylor", net_params=net_params)
    # name = f"naylor_reunet_channels{net_params['base_channels']}_factor{net_params['base_channels_factor']}_aspp{net_params['aspp_inter_channels']}_lr=reu200"
    # train(net=net, project="reu_net", name=name, train_params=train_params, pprocess_params=None)
    # net = REUNet(mode="graham", net_params=net_params)
    # name = f"graham_reunet_channels{net_params['base_channels']}_factor{net_params['base_channels_factor']}_aspp{net_params['aspp_inter_channels']}_lr=reu200"
    # train(net=net, project="reu_net", name=name, train_params=train_params, pprocess_params=None)
    #
    # net_params = {
    #     "aspp_inter_channels": 80,
    #     "base_channels": 80,
    #     "base_channels_factor": 2
    # }
    # net = REUNet(mode="yang", net_params=net_params)
    # name = f"yang_reunet_channels{net_params['base_channels']}_factor{net_params['base_channels_factor']}_aspp{net_params['aspp_inter_channels']}_lr=reu200"
    # train(net=net, project="reu_net", name=name, train_params=train_params, pprocess_params=None)
    # net = REUNet(mode="naylor", net_params=net_params)
    # name = f"naylor_reunet_channels{net_params['base_channels']}_factor{net_params['base_channels_factor']}_aspp{net_params['aspp_inter_channels']}_lr=reu200"
    # train(net=net, project="reu_net", name=name, train_params=train_params, pprocess_params=None)
    # net = REUNet(mode="graham", net_params=net_params)
    # name = f"graham_reunet_channels{net_params['base_channels']}_factor{net_params['base_channels_factor']}_aspp{net_params['aspp_inter_channels']}_lr=reu200"
    # train(net=net, project="reu_net", name=name, train_params=train_params, pprocess_params=None)
    #
    # net_params = {
    #     "base_channels": 32,
    #     "aspp_inter_channels": 16,
    #     "base_channels_factor": 1,
    # }
    # net = REUNet(mode="yang", net_params=net_params)
    # name = f"yang_reunet_channels{net_params['base_channels']}_factor{net_params['base_channels_factor']}_aspp{net_params['aspp_inter_channels']}_lr=reu200"
    # train(net=net, project="reu_net", name=name, train_params=train_params, pprocess_params=None)
    # net = REUNet(mode="naylor", net_params=net_params)
    # name = f"naylor_reunet_channels{net_params['base_channels']}_factor{net_params['base_channels_factor']}_aspp{net_params['aspp_inter_channels']}_lr=reu200"
    # train(net=net, project="reu_net", name=name, train_params=train_params, pprocess_params=None)
    # net = REUNet(mode="graham", net_params=net_params)
    # name = f"graham_reunet_channels{net_params['base_channels']}_factor{net_params['base_channels_factor']}_aspp{net_params['aspp_inter_channels']}_lr=reu200"
    # train(net=net, project="reu_net", name=name, train_params=train_params, pprocess_params=None)
    #
    # net_params = {
    #     "base_channels": 32,
    #     "aspp_inter_channels": 16,
    #     "base_channels_factor": 2,
    # }
    # net = REUNet(mode="yang", net_params=net_params)
    # name = f"yang_reunet_channels{net_params['base_channels']}_factor{net_params['base_channels_factor']}_aspp{net_params['aspp_inter_channels']}_lr=reu200"
    # train(net=net, project="reu_net", name=name, train_params=train_params, pprocess_params=None)
    # net = REUNet(mode="naylor", net_params=net_params)
    # name = f"naylor_reunet_channels{net_params['base_channels']}_factor{net_params['base_channels_factor']}_aspp{net_params['aspp_inter_channels']}_lr=reu200"
    # train(net=net, project="reu_net", name=name, train_params=train_params, pprocess_params=None)
    # net = REUNet(mode="graham", net_params=net_params)
    # name = f"graham_reunet_channels{net_params['base_channels']}_factor{net_params['base_channels_factor']}_aspp{net_params['aspp_inter_channels']}_lr=reu200"
    # train(net=net, project="reu_net", name=name, train_params=train_params, pprocess_params=None)
    #
    # net_params = {
    #     "base_channels": 32,
    #     "aspp_inter_channels": 32,
    #     "base_channels_factor": 1,
    # }
    # net = REUNet(mode="yang", net_params=net_params)
    # name = f"yang_reunet_channels{net_params['base_channels']}_factor{net_params['base_channels_factor']}_aspp{net_params['aspp_inter_channels']}_lr=reu200"
    # train(net=net, project="reu_net", name=name, train_params=train_params, pprocess_params=None)
    # net = REUNet(mode="naylor", net_params=net_params)
    # name = f"naylor_reunet_channels{net_params['base_channels']}_factor{net_params['base_channels_factor']}_aspp{net_params['aspp_inter_channels']}_lr=reu200"
    # train(net=net, project="reu_net", name=name, train_params=train_params, pprocess_params=None)
    # net = REUNet(mode="graham", net_params=net_params)
    # name = f"graham_reunet_channels{net_params['base_channels']}_factor{net_params['base_channels_factor']}_aspp{net_params['aspp_inter_channels']}_lr=reu200"
    # train(net=net, project="reu_net", name=name, train_params=train_params, pprocess_params=None)

    ########################################################################################################

    # net_params = {
    #     "aspp": False,
    # }
    # net = ALNetLight(mode="yang", net_params=net_params)
    # name = f"yang_alnetlight_lr=reu200"
    # train(net=net, project="al_net", name=name, train_params=train_params, pprocess_params=None)
    # net = ALNetLight(mode="naylor", net_params=net_params)
    # name = f"naylor_alnetlight_lr=reu200"
    # train(net=net, project="al_net", name=name, train_params=train_params, pprocess_params=None)
    # net = ALNetLight(mode="graham", net_params=net_params)
    # name = f"graham_alnetlight_lr=reu200"
    # train(net=net, project="al_net", name=name, train_params=train_params, pprocess_params=None)
    #
    # net_params = {
    #     "aspp": True,
    #     "aspp_inter_channels": 64,
    # }
    # net = ALNetLight(mode="yang", net_params=net_params)
    # name = f"yang_alnetlight_aspp{net_params['aspp_inter_channels']}_lr=reu200"
    # train(net=net, project="al_net", name=name, train_params=train_params, pprocess_params=None)
    # net = ALNetLight(mode="naylor", net_params=net_params)
    # name = f"naylor_alnetlight_aspp{net_params['aspp_inter_channels']}_lr=reu200"
    # train(net=net, project="al_net", name=name, train_params=train_params, pprocess_params=None)
    # net = ALNetLight(mode="graham", net_params=net_params)
    # name = f"graham_alnetlight_aspp{net_params['aspp_inter_channels']}_lr=reu200"
    # train(net=net, project="al_net", name=name, train_params=train_params, pprocess_params=None)

    # net = ALNetDualDecoder(mode="yang")
    # name = "yang_alnetdual_lr=reu200"
    # train(net=net, project="al_net", name=name, train_params=train_params, pprocess_params=None)
    # net = ALNetDualDecoder(mode="naylor")
    # name = "naylor_alnetdual_lr=reu200"
    # train(net=net, project="al_net", name=name, train_params=train_params, pprocess_params=None)
    # net = ALNetDualDecoder(mode="graham")
    # name = "graham_alnetdual_lr=reu200"
    # train(net=net, project="al_net", name=name, train_params=train_params, pprocess_params=None)

    # net_params = {
    #     "aspp_inter_channels": 96,
    #     "base_channels": 96,
    #     "base_channels_factor": 2
    # }
    # train_params.update({"batch_size": 6})
    # net = REUNet(mode="yang", net_params=net_params)
    # name = f"yang_reunet_channels{net_params['base_channels']}_factor{net_params['base_channels_factor']}_aspp{net_params['aspp_inter_channels']}_lr=reu200"
    # train(net=net, project="reu_net", name=name, train_params=train_params, pprocess_params=None)
    # net = REUNet(mode="naylor", net_params=net_params)
    # name = f"naylor_reunet_channels{net_params['base_channels']}_factor{net_params['base_channels_factor']}_aspp{net_params['aspp_inter_channels']}_lr=reu200"
    # train(net=net, project="reu_net", name=name, train_params=train_params, pprocess_params=None)
    # net = REUNet(mode="graham", net_params=net_params)
    # name = f"graham_reunet_channels{net_params['base_channels']}_factor{net_params['base_channels_factor']}_aspp{net_params['aspp_inter_channels']}_lr=reu200"
    # train(net=net, project="reu_net", name=name, train_params=train_params, pprocess_params=None)

    # net_params = {
    #     "aspp": False,
    # }
    # net = ALNetLightDualDecoder(mode="yang", net_params=net_params)
    # name = f"yang_alnetlightdual_lr=reu200"
    # train(net=net, project="al_net", name=name, train_params=train_params, pprocess_params=None)
    # net = ALNetLightDualDecoder(mode="naylor", net_params=net_params)
    # name = f"naylor_alnetlightdual_lr=reu200"
    # train(net=net, project="al_net", name=name, train_params=train_params, pprocess_params=None)
    # net = ALNetLightDualDecoder(mode="graham", net_params=net_params)
    # name = f"graham_alnetlightdual_lr=reu200"
    # train(net=net, project="al_net", name=name, train_params=train_params, pprocess_params=None)
    #
    # net_params = {
    #     "base_channels": 32,
    #     "aspp_inter_channels": 32,
    #     "base_channels_factor": 2
    # }
    # net = REUNet(mode="yang", net_params=net_params)
    # name = f"yang_reunet_channels{net_params['base_channels']}_factor{net_params['base_channels_factor']}_aspp{net_params['aspp_inter_channels']}_lr=reu200"
    # train(net=net, project="reu_net", name=name, train_params=train_params, pprocess_params=None)
    # net = REUNet(mode="naylor", net_params=net_params)
    # name = f"naylor_reunet_channels{net_params['base_channels']}_factor{net_params['base_channels_factor']}_aspp{net_params['aspp_inter_channels']}_lr=reu200"
    # train(net=net, project="reu_net", name=name, train_params=train_params, pprocess_params=None)
    # net = REUNet(mode="graham", net_params=net_params)
    # name = f"graham_reunet_channels{net_params['base_channels']}_factor{net_params['base_channels_factor']}_aspp{net_params['aspp_inter_channels']}_lr=reu200"
    # train(net=net, project="reu_net", name=name, train_params=train_params, pprocess_params=None)

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
    # pl.seed_everything(42, workers=True)  # Seed for torch, numpy and python.random
    # net = UNet(mode="baseline")
    # name = f"baseline_unet_lr=cpp200"
    # train(net=net, project="u_net", name=name, train_params=train_params, pprocess_params=None)
    #
    # pl.seed_everything(42, workers=True)  # Seed for torch, numpy and python.random
    # net = UNet(mode="baseline")
    # name = f"baseline_unet_lr=cpp200"
    # train(net=net, project="u_net", name=name, train_params=train_params, pprocess_params=None)
    pl.seed_everything(42, workers=True)  # Seed for torch, numpy and python.random
    net = UNet(mode="noname")
    name = f"noname_unet_lr=cpp200"
    train(net=net, project="u_net", name=name, train_params=train_params, pprocess_params=None)
    pl.seed_everything(42, workers=True)  # Seed for torch, numpy and python.random
    net = UNet(mode="naylor")
    name = f"naylor_unet_lr=cpp200"
    train(net=net, project="u_net", name=name, train_params=train_params, pprocess_params=None)
    pl.seed_everything(42, workers=True)  # Seed for torch, numpy and python.random
    net = UNet(mode="graham")
    name = f"graham_unet_lr=cpp200"
    train(net=net, project="u_net", name=name, train_params=train_params, pprocess_params=None)

    # pl.seed_everything(42, workers=True)  # Seed for torch, numpy and python.random
    # net = UNetDualDecoder(mode="noname")
    # name = f"noname_unetdual_lr=cpp200"
    # train(net=net, project="u_net_dual", name=name, train_params=train_params, pprocess_params=None)
    # pl.seed_everything(42, workers=True)  # Seed for torch, numpy and python.random
    # net = UNetDualDecoder(mode="graham")
    # name = f"graham_unetdual_lr=cpp200"
    # train(net=net, project="u_net_dual", name=name, train_params=train_params, pprocess_params=None)
