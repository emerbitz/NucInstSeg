import unittest
import torch
from torch import Tensor

from postprocessing.region_growing import RegionGrower
from postprocessing.utils import assert_equal


class TestRegionGrower(unittest.TestCase):
    def setUp(self) -> None:
        self.addTypeEqualityFunc(Tensor, assert_equal)

    def test_0(self):
        mask = torch.tensor([[[1, 0, 0], [0, 0, 0], [0, 0, 1]]], dtype=torch.bool)
        gt_inst = torch.tensor([[[1, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 1]]], dtype=torch.bool)
        self.assertEqual(RegionGrower(mask).get_regions(), gt_inst)

    def test_1(self):
        mask = torch.tensor([[[1, 1, 1, 0, 0, 0],
                              [1, 1, 0, 0, 0, 0],
                              [1, 0, 0, 0, 0, 1],
                              [0, 0, 1, 1, 1, 0],
                              [0, 1, 1, 0, 1, 0],
                              [0, 0, 1, 0, 1, 0]]], dtype=torch.bool)

        gt_inst = torch.tensor([[[1, 1, 1, 0, 0, 0],
                                 [1, 1, 0, 0, 0, 0],
                                 [1, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0]],

                                [[0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 1],
                                 [0, 0, 1, 1, 1, 0],
                                 [0, 1, 1, 0, 1, 0],
                                 [0, 0, 1, 0, 1, 0]]], dtype=torch.bool)
        self.assertEqual(RegionGrower(mask).get_regions(), gt_inst)