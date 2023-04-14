import unittest
import torch
from torch import Tensor

from postprocessing.segmentation import InstanceExtractor
from postprocessing.utils import assert_equal


class TestInstanceExtractor(unittest.TestCase):
    def setUp(self) -> None:
        self.addTypeEqualityFunc(Tensor, assert_equal)

    def test_0(self):
        seg = torch.tensor([[[[0.49, 0.50], [0.51, 0.52]]]], dtype=torch.float)
        cont = torch.tensor([[[[0, 1], [1, 0]]]], dtype=torch.float)
        extractor = InstanceExtractor(seg, cont)
        with self.subTest(msg="Mask"):
            self.assertEqual(extractor.mask, torch.tensor([[[[0, 0], [0, 1]]]], dtype=torch.bool))
        with self.subTest(msg="Batch size"):
            self.assertEqual(extractor.batch_size, 1)
        instances = extractor.get_instances(impl="custom")
        with self.subTest(msg="Instances"):
            self.assertEqual(instances, torch.tensor([[[0, 0], [0, 1]]], dtype=torch.bool))

    def test_1(self):
        seg = torch.tensor([[[[0, 0], [1, 1]]]], dtype=torch.float)
        extractor = InstanceExtractor(seg)
        with self.subTest(msg="Mask"):
            self.assertEqual(extractor.mask, torch.tensor([[[[0, 0], [1, 1]]]], dtype=torch.bool))
        instances = extractor.get_instances(impl="custom")
        print(instances)
        with self.subTest(msg="Instances"):
            self.assertEqual(instances, torch.tensor([[[0, 0], [1, 1]]], dtype=torch.bool))


if __name__ == '__main__':
    unittest.main()
