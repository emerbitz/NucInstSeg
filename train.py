import pytorch_lightning as pl
from pytorch_lightning.profilers import SimpleProfiler

from data.MoNuSeg.data_module import MoNuSegDataModule
from models.al_net import ALNet
from models.al_net_model import ALNetModel


def main():
    pl.seed_everything(42, workers=True)  # Seed for torch, numpy and python.random
    net = ALNet(up_mode="nearest")
    model = ALNetModel(net)
    data_module = MoNuSegDataModule(seg_masks=True, cont_masks=True, dist_maps=False)
    logger = pl.loggers.TensorBoardLogger(save_dir=".", default_hp_metric=False)
    # Alternative: SimpleProfiler(), AdvancedProfiler(), XLAProfiler()
    profiler = SimpleProfiler(dirpath="performance_logs", filename="summary")

    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=1, max_time="0:8:0:0", auto_lr_find=False,
                         auto_scale_batch_size=False, logger=logger, deterministic=False, fast_dev_run=False,
                         profiler=profiler, log_every_n_steps=1, check_val_every_n_epoch=1, enable_model_summary=True,
                         overfit_batches=0.0,
                         limit_train_batches=None, limit_val_batches=None, limit_test_batches=None,
                         )
    # callbacks=[ModelSummary(max_depth=-1)]
    # callbacks = [DeviceStatsMonitor()]
    # trainer.tune(model, data_module)  # for automatic selection of lr and batch size
    trainer.fit(model, data_module)
    # trainer.test(model, data_module)

    # trainer = pl.Trainer(accelerator="gpu", devices=1)
    # lr_finder = trainer.tuner.lr_find(model, data_module)
    # # print(lr_finder.results)
    # lr = lr_finder.suggestion()
    # print(lr)
    # fig = lr_finder.plot(suggest=True)
    # fig.show()


if __name__ == "__main__":
    main()
