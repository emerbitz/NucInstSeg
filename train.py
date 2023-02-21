import pytorch_lightning as pl

from data.MoNuSeg.data_module import MoNuSegDataModule

def main():
    model = None # Dummy
    data_module = MoNuSegDataModule(seg_masks=True, cont_masks=True, dist_maps=True)

    trainer = pl.Trainer()
    trainer.fit(model, data_module)
    trainer.test(model, data_module)

if __name__ == "__main__":
    pass