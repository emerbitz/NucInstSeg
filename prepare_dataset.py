from data.MoNuSeg.data_module import MoNuSegDataModule

def main():
    data_module = MoNuSegDataModule(
        seg_masks=True,
        cont_masks=True,
        dist_maps=True,
        hv_maps=True,
        labels=True,
        img_size=(256, 256),
        data_root="datasets"
    )
    data_module.prepare_data()

if __name__ == "__main__":
    main()