import pytorch_lightning as pl

from data.MoNuSeg.data_module import MoNuSegDataModule

from models.u_net import UNet
from models.u_net_model import UNetModel


def main():
    net = UNet(in_channels=3, out_channels=1, feature_num=16)
    model = UNetModel(net)
    data_module = MoNuSegDataModule(seg_masks=True, cont_masks=False, dist_maps=False)

    trainer = pl.Trainer(gpus=1)
    trainer.fit(model, data_module)
    trainer.test(model, data_module)


if __name__ == "__main__":
    main()
