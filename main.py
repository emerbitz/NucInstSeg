from data.MoNuSeg.dataset_creator import MoNuSegCreator


def main():
    MoNuSegCreator(root="datasets").save_ground_truths()


if __name__ == '__main__':
    main()

