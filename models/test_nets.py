import unittest
from pathlib import Path
from typing import Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from data.MoNuSeg.data_module import MoNuSegDataModule
from models.al_net_light import ALNetLightDualDecoder
from models.net_model import NetModel
from models.utils import rm_tree


class TestNet(unittest.TestCase):
    def setUp(self, net=ALNetLightDualDecoder(mode="naylor", net_params={"auxiliary_task": True, "aspp": True,
                                                                         "assp_inter_channels": 32})) -> None:
        log_base_dir = Path("../lightning_logs")
        log_name = "testing"
        self.log_dir = Path(log_base_dir, log_name, "version_0")
        ckpt_dir = Path("../trained_models/testing")
        self.ckpt_file = Path(ckpt_dir, "last.ckpt")

        self.trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=1, limit_train_batches=2,
                                  num_sanity_val_steps=0, check_val_every_n_epoch=1,
                                  limit_val_batches=1, enable_checkpointing=False,
                                  logger=TensorBoardLogger(save_dir=log_base_dir, name=log_name,
                                                           default_hp_metric=False),
                                  # callbacks=[ModelCheckpoint(dirpath=ckpt_dir, save_top_k=0, save_last=True,
                                  #                            save_weights_only=True)],
                                  enable_model_summary=False, enable_progress_bar=False)
        pl.seed_everything(42, workers=True)
        self.net = net
        self.model = NetModel(net)
        self.data_module = MoNuSegDataModule.default_mode(net.mode, auxiliary_task=self.model.auxiliary_task,
                                                          batch_size=8, data_root="../datasets")
        self.state_initial = self.model.state_dict()

    def test_all_subtests(self):
        for subtest_name, subtest in self._get_subtests():
            try:
                subtest()
            except Exception as e:
                self.fail(f"{subtest_name} failed with {type(e)}:{e}")

    def tearDown(self) -> None:
        # Remove log dir and ckpt-file:
        rm_tree(self.log_dir)
        # self.ckpt_file.unlink()

    def _get_subtests(self) -> Tuple[str]:
        for attr_name in dir(TestNet):
            if attr_name.startswith("subtest"):
                yield attr_name, getattr(self, attr_name)

    def subtest_0_runable(self):
        with self.subTest(msg="Subtest: Runable"):
            self.assertIs(self.trainer.fit(model=self.model, datamodule=self.data_module), None)

    def subtest_1_all_param_trainable(self):
        # state_initial = self.model.state_dict()
        state_initial = self.state_initial
        # state_final = torch.load(self.ckpt_file)["state_dict"]
        state_final = self.model.state_dict()
        non_trained = []
        for state_name in state_initial.keys():
            was_trained = torch.any(state_initial[state_name] != state_final[state_name].cpu())
            if not was_trained:
                non_trained.append(state_name)
        with self.subTest(msg="Subtest: All param trainable"):
            self.assertEqual(len(non_trained), 0, msg=f"The following state/s was/were not trained:\n{non_trained}")

    def subtest_2_valid_loss(self):
        with self.subTest(msg="Subtest: Valid train loss"):
            train_loss = self.trainer.logged_metrics["train_loss"]
            self.assertGreater(train_loss, 0, msg=f"Invalid train loss of {train_loss}.")
        with self.subTest(msg="Subtest: Valid val loss"):
            val_loss = self.trainer.logged_metrics["val_loss"]
            self.assertGreater(train_loss, 0, msg=f"Invalid train loss of {val_loss}.")
