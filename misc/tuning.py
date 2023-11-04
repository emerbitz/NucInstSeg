from collections import defaultdict

import pytorch_lightning as pl
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.loggers import TensorBoardLogger

from data.MoNuSeg.data_module import MoNuSegDataModule
from models.al_net import ALNet
from models.net_module import NetModule
from models.reu_net import REUNet


def suggest_train_params(trial):
    train_params = defaultdict()
    train_params["max_epochs"] = 200
    train_params["batch_size"] = 8

    train_params["lr_schedule"] = trial.suggest_categorical("lr_schedule", ["none", "plateau", "warmup_cos",
                                                                            "cos_warm_restart"])
    if train_params["lr_schedule"] == "none":
        train_params["lr"] = trial.suggest_float("lr_none", 1e-4, 1e-3, log=False)

    elif train_params["lr_schedule"] == "plateau":
        train_params["lr"] = trial.suggest_float("lr_plateau", 1e-4, 1e-3, log=False)
        train_params["min_lr"] = trial.suggest_float("min_lr_plateau", 1e-8, 1e-4, log=False)
        train_params["reduction_factor"] = trial.suggest_categorical("reduction_factor", [0.5, 0.25, 0.1])
        train_params["patience"] = trial.suggest_categorical("patience", [5, 10, 15])

    elif train_params["lr_schedule"] == "warmup_cos":
        train_params["lr"] = trial.suggest_float("lr_cos", 1e-4, 1e-3, log=False)
        train_params["min_lr"] = trial.suggest_float("min_lr_cos", 0, 1e-4, log=False)
        train_params["warmup_epochs"] = trial.suggest_int("warmup_epochs", 1, 5)

    elif train_params["lr_schedule"] == "cos_warm_restart":
        train_params["lr"] = trial.suggest_float("lr_restart", 1e-4, 1e-3, log=False)
        train_params["min_lr"] = trial.suggest_float("min_lr_restart", 0, 1e-4, log=False)
        train_params["period_mult"] = trial.suggest_int("period_mult", 1, 4)
        period_mult_as_str = str(train_params["period_mult"])
        period_initial = {"1": [100, 50, 25], "2": [67, 29, 13], "3": [50, 15], "4": [40, 10]}
        train_params["period_initial"] = trial.suggest_categorical("period_initial", period_initial[period_mult_as_str])

    return train_params


def tune_train_params(trial):
    mode = "naylor"
    net = ALNet(mode=mode)

    pl.seed_everything(42, workers=True)  # Seed for torch, numpy and python.random
    train_params = suggest_train_params(trial)
    log_name = str(train_params)
    model = NetModule(net=net, train_params=train_params, pprocess_params=None)
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, max_epochs=model.train_params["max_epochs"], max_time="0:3:0:0",
        fast_dev_run=False, log_every_n_steps=1, check_val_every_n_epoch=1, enable_model_summary=False,
        enable_checkpointing=False, callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
        logger=TensorBoardLogger(save_dir="lightning_logs/tuning", name=log_name, default_hp_metric=False),
    )

    data_module = MoNuSegDataModule.default_mode(mode=model.mode, auxiliary_task=model.auxiliary_task,
                                                 batch_size=model.train_params["batch_size"])
    trainer.fit(model, data_module)
    return trainer.logged_metrics["val_loss"]


def tune_net_params(trial):
    mode = "contour"

    pl.seed_everything(42, workers=True)  # Seed for torch, numpy and python.random
    net_params = {
        "inter_channels_equal_out_channels": True,
        "aspp_inter_channels": trial.suggest_categorical("aspp_inter_channels", [1, 4, 8, 12, 16]),
        "norm_actv": trial.suggest_categorical("norm_actv", [True, False]),
        "down_mode": trial.suggest_categorical("down_mode", ["max", "avg", "conv"])
    }
    log_name = f"{mode}_reu_net_params_assp{net_params['aspp_inter_channels']}_dmode{net_params['down_mode']}_notv{net_params['norm_actv']}"
    model = NetModule(net=REUNet(mode=mode, net_params=net_params))
    log_folder = f"{mode}_reu_net_params"
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, max_epochs=model.train_params["max_epochs"], max_time="0:8:0:0",
        fast_dev_run=True, log_every_n_steps=1, check_val_every_n_epoch=1, enable_model_summary=False,
        enable_checkpointing=False, callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
        logger=TensorBoardLogger(save_dir=f"lightning_logs/tuning/{log_folder}", name=log_name,
                                 default_hp_metric=False),
    )
    data_module = MoNuSegDataModule.default_mode(mode=model.mode, auxiliary_task=model.auxiliary_task,
                                                 batch_size=model.train_params["batch_size"])
    trainer.fit(model, data_module)
    return trainer.logged_metrics["val_loss"]


if __name__ == '__main__':
    pass
    # study = optuna.create_study(direction="maximize")
    # study.optimize(tune_pprocess_params, n_trials=100)
    # # study = optuna.create_study(direction="minimize", pruner=MedianPruner())
    # # study.optimize(tune_train_params, n_trials=100)
    #
    # print("Number of finished trials: ", len(study.trials))
    # print("Best trial:")
    # trial = study.best_trial
    #
    # print("Value: ", trial.value)
    # print("Params: ")
    # for key, value in trial.params.items():
    #     print(f"    {key}: {value}")
