from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import optuna
import pytorch_lightning as pl
from optuna.samplers import TPESampler
from pytorch_lightning.loggers import TensorBoardLogger

from data.MoNuSeg.data_module import MoNuSegDataModule
from models.net_module import NetModule
from models.reu_net import REUNet


def suggest_thresh(trial, thresh_name: str, is_float: bool = True, range_float: Optional[Tuple[float, float]] = None):
    """Suggests a threshold value or a threshold selection algorithm."""

    if trial.suggest_categorical(f"{thresh_name}_is_scalar", [True, False]):
        if is_float:
            if range_float is None:
                range_float = (0.10, 0.8)
            thresh = trial.suggest_float(f"{thresh_name}_float", range_float[0], range_float[1], step=0.05)
        else:
            thresh = trial.suggest_int(f"{thresh_name}_int", 0, 2, step=1)
    else:
        thresh = trial.suggest_categorical(f"{thresh_name}_cat", ["otsu", "yen", "isodata", "li", "mean", "triangle"])

    return thresh


def suggest_pprocess_params(trial, mode: str) -> defaultdict:
    """Suggests postprocessing parameters for the given postprocessing method."""

    pprocess_params = defaultdict()

    if mode == "baseline":
        pprocess_params["thresh_seg"] = suggest_thresh(trial, thresh_name="thresh_seg")
        pprocess_params["min_obj_size"] = trial.suggest_int("min_obj_size", 0, 80, step=5)

    elif mode == "contour":
        pprocess_params["thresh_seg"] = suggest_thresh(trial, thresh_name="thresh_seg")
        pprocess_params["thresh_cont"] = suggest_thresh(trial, thresh_name="thresh_cont")
        pprocess_params["min_obj_size"] = trial.suggest_int("min_obj_size", 0, 80, step=5)
        pprocess_params["min_marker_size"] = trial.suggest_int("min_marker_size", 0, 80, step=5)

    elif mode == "yang":
        pprocess_params["thresh_seg"] = suggest_thresh(trial, thresh_name="thresh_seg")
        pprocess_params["min_obj_size"] = trial.suggest_int("min_obj_size", 0, 80, step=5)
        pprocess_params["thresh_coarse"] = trial.suggest_int("thresh_coarse", 0, 500, step=10)
        pprocess_params["thresh_fine"] = trial.suggest_int("thresh_fine", 0, pprocess_params["thresh_coarse"], step=10)

    elif mode == "naylor":
        pprocess_params["thresh_dist"] = suggest_thresh(trial, thresh_name="dist", range_float=(0.3, 1.8), is_float=True)
        # pprocess_params["thresh_dist"] = suggest_thresh(trial, thresh_name="dist", is_float=False)
        pprocess_params["h_param"] = trial.suggest_int("h_param", 0, 30, step=1)

    elif mode == "graham":
        pprocess_params["thresh_seg"] = suggest_thresh(trial, thresh_name="thresh_seg")
        pprocess_params["min_obj_size"] = trial.suggest_int("min_obj_size", 0, 80, step=5)
        pprocess_params["thresh_comb"] = suggest_thresh(trial, thresh_name="thresh_comb")
        pprocess_params["min_marker_size"] = trial.suggest_int("min_marker_size", 0, 80, step=5)

    elif mode == "exprmtl":
        pprocess_params["thresh_seg"] = suggest_thresh(trial, thresh_name="thresh_seg")
        pprocess_params["min_obj_size"] = trial.suggest_int("min_obj_size", 0, 80, step=5)
        pprocess_params["thresh_comb"] = suggest_thresh(trial, thresh_name="thresh_comb")
        pprocess_params["min_marker_size"] = trial.suggest_int("min_marker_size", 0, 80, step=5)

    return pprocess_params


class PostProcessParamsTuner:
    """
    Tunes the postprocessing parameters for a given postprocesssing method.
    """
    def __init__(self, net, project: str, ckpt_name: str, tune_from: str = "val",
                 objective: str = "PQ", direction: str = "maximize", n_trials: int = 100, seed: int = 40):

        self.net = net
        self.mode = net.mode
        self.project = project
        self.ckpt_name = ckpt_name
        self.tune_from = tune_from
        self.log_dir = Path("lightning_logs", "tuning", "pprocess_params", f"{self.project}_{self.mode}")
        self.ckpt_file = self.get_ckpt_file()
        self.objective = objective
        self.direction = direction
        self.n_trials = n_trials
        self.seed = seed

    def __call__(self) -> Tuple[Dict[str, Any], str]:
        sampler = TPESampler(seed=self.seed)
        study = optuna.create_study(sampler=sampler, direction=self.direction)
        study.optimize(self.tune_pprocess_params, n_trials=self.n_trials)

        trial = study.best_trial

        print("Value: ", trial.value)
        print("Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        return trial.params, self.ckpt_file

    def tune_pprocess_params(self, trial):
        pprocess_params = suggest_pprocess_params(trial, mode=self.mode)
        ############################
        log_name = str(pprocess_params)
        pl.seed_everything(42, workers=True)  # Seed for torch, numpy and python.random
        trainer = pl.Trainer(
            accelerator="gpu", devices=1,
            logger=TensorBoardLogger(save_dir=self.log_dir, name=log_name, default_hp_metric=False)
        )
        model = NetModule(net=self.net, pprocess_params=pprocess_params)
        data_module = MoNuSegDataModule.default_mode(mode=self.mode, auxiliary_task=model.auxiliary_task,
                                                     batch_size=model.train_params["batch_size"],
                                                     test_data_is_val_data=True)
        trainer.test(model, data_module, ckpt_path=self.ckpt_file, verbose=False)
        ###############
        return trainer.logged_metrics[self.objective]

    def get_ckpt_file(self) -> str:
        """Retrieves the name of the checkpoint (ckpt) file."""
        ckpt_dir = Path("trained_models", self.project, self.ckpt_name)
        ckpt_file = ckpt_dir.glob(f"*{self.tune_from}*.ckpt")
        ckpt_file = list(ckpt_file)
        if len(ckpt_file) > 1:
            raise ValueError(f"Found multiple checkpoint files: {ckpt_file}.")
        if not len(ckpt_file):
            raise ValueError(f"Found no checkpoint file in the directory '{ckpt_dir}'.")
        return str(ckpt_file[0])


if __name__ == '__main__':
    #######################################################################################
    # Example for the tuning of the postprocessing parameters in combination with a REU-Net
    #######################################################################################
    net_params = {
        "aspp_inter_channels": 96,
        "base_channels": 48,
        "base_channels_factor": 2,
        "depth": 4,
        "norm": False
    }
    # Tuning of the baseline postprocessing method
    net = REUNet(mode="baseline", net_params=net_params)
    name = "baseline_reunet_netchannels48_factor2_aspp96_depth4_lr=reu200"
    project = "reu_net"
    pparams, ckpt_file = PostProcessParamsTuner(net=net, project=project, ckpt_name=name, n_trials=100)()
    # # Tuning of the Yang postprocessing method
    # net = REUNet(mode="yang", net_params=net_params)
    # name = "baseline_reunet_netchannels48_factor2_aspp96_depth4_lr=reu200"
    # project = "reu_net"
    # pparams, ckpt_file = PostProcessParamsTuner(net=net, project=project, ckpt_name=name, n_trials=100)()
    # # Tuning of the contour-based postprocessing method
    # net = REUNet(mode="contour", net_params=net_params)
    # name = "baseline_reunet_netchannels48_factor2_aspp96_depth4_lr=reu200"
    # project = "reu_net"
    # pparams, ckpt_file = PostProcessParamsTuner(net=net, project=project, ckpt_name=name, n_trials=100)()
    # # Tuning of the Graham postprocessing method
    # net = REUNet(mode="graham", net_params=net_params)
    # name = "graham_reunet_netchannels48_factor2_aspp96_depth4_lr=reu200"
    # project = "reu_net"
    # pparams, ckpt_file = PostProcessParamsTuner(net=net, project=project, ckpt_name=name, n_trials=100)()
    # # Tuning of the Graham derived postprocessing method
    # net = REUNet(mode="exprmtl", net_params=net_params)
    # name = "graham_reunet_netchannels48_factor2_aspp96_depth4_lr=reu200"
    # project = "reu_net"
    # pparams, ckpt_file = PostProcessParamsTuner(net=net, project=project, ckpt_name=name, n_trials=100)()
