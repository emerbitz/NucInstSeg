import unittest
import numpy as np
import torch

from evaluation.metrics import PQ, AJI, ModAJI


class TestPQ(unittest.TestCase):
    """Unite test case for the PQ class."""

    def setUp(self) -> None:
        self.metric = PQ()

    def test_0(self):
        pred = torch.tensor([[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                             [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool)

        target = torch.tensor([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool)
        dq, sq, pq = self.metric(pred, target).values()
        with self.subTest(msg="TP"):
            self.assertEqual(self.metric.TP, torch.tensor(0, dtype=torch.int))
        with self.subTest(msg="FN"):
            self.assertEqual(self.metric.FN, torch.tensor(2, dtype=torch.int))
        with self.subTest(msg="FP"):
            self.assertEqual(self.metric.FP, torch.tensor(2, dtype=torch.int))
        with self.subTest(msg="IoU"):
            self.assertEqual(self.metric.IoU, torch.tensor(0, dtype=torch.float))
        with self.subTest(msg="DQ"):
            self.assertEqual(dq, torch.tensor(0, dtype=torch.float))
        with self.subTest(msg="SQ"):
            self.assertTrue(sq.isnan())
        with self.subTest(msg="PQ"):
            self.assertTrue(pq.isnan())

    def test_1(self):
        pred = torch.tensor([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                              [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                              [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                             [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool)

        target = torch.tensor([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool)
        dq, sq, pq = self.metric(pred, target).values()
        with self.subTest(msg="TP"):
            self.assertEqual(self.metric.TP, torch.tensor(1, dtype=torch.int))
        with self.subTest(msg="FN"):
            self.assertEqual(self.metric.FN, torch.tensor(1, dtype=torch.int))
        with self.subTest(msg="FP"):
            self.assertEqual(self.metric.FP, torch.tensor(1, dtype=torch.int))
        with self.subTest(msg="IoU"):
            self.assertEqual(self.metric.IoU, torch.tensor(1, dtype=torch.float))
        with self.subTest(msg="DQ"):
            self.assertEqual(dq, torch.tensor(0.5, dtype=torch.float))
        with self.subTest(msg="SQ"):
            self.assertEqual(sq, torch.tensor(1, dtype=torch.float))
        with self.subTest(msg="PQ"):
            self.assertEqual(pq, torch.tensor(0.5, dtype=torch.float))

    def test_2(self):
        pred = torch.tensor([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                             [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool)

        target = torch.tensor([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool)
        dq, sq, pq = self.metric(pred, target).values()
        with self.subTest(msg="TP"):
            self.assertEqual(self.metric.TP, torch.tensor(1, dtype=torch.int))
        with self.subTest(msg="FN"):
            self.assertEqual(self.metric.FN, torch.tensor(1, dtype=torch.int))
        with self.subTest(msg="FP"):
            self.assertEqual(self.metric.FP, torch.tensor(1, dtype=torch.int))
        with self.subTest(msg="IoU"):
            self.assertEqual(self.metric.IoU, torch.tensor(0.7, dtype=torch.float))
        with self.subTest(msg="DQ"):
            self.assertEqual(dq, torch.tensor(0.5, dtype=torch.float))
        with self.subTest(msg="SQ"):
            self.assertEqual(sq, torch.tensor(0.7, dtype=torch.float))
        with self.subTest(msg="PQ"):
            self.assertEqual(pq, torch.tensor(0.35, dtype=torch.float))

    def test_3(self):
        pred = torch.tensor([[[0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                             [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool)

        target = torch.tensor([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
                                [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool)
        dq, sq, pq = self.metric(pred, target).values()
        with self.subTest(msg="TP"):
            self.assertEqual(self.metric.TP, torch.tensor(1, dtype=torch.int))
        with self.subTest(msg="FN"):
            self.assertEqual(self.metric.FN, torch.tensor(2, dtype=torch.int))
        with self.subTest(msg="FP"):
            self.assertEqual(self.metric.FP, torch.tensor(1, dtype=torch.int))
        with self.subTest(msg="IoU"):
            self.assertEqual(self.metric.IoU, torch.tensor(7 / 12, dtype=torch.float))
        with self.subTest(msg="DQ"):
            self.assertEqual(dq, torch.tensor(0.4, dtype=torch.float))
        with self.subTest(msg="SQ"):
            self.assertEqual(sq, torch.tensor(7 / 12, dtype=torch.float))
        with self.subTest(msg="PQ"):
            self.assertEqual(pq, torch.tensor(0.4, dtype=torch.float) * torch.tensor(7 / 12, dtype=torch.float))

    def test_4(self):
        pred = torch.tensor([[[0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                             [[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool)

        target = torch.tensor([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool)

        dq, sq, pq = self.metric(pred, target).values()
        with self.subTest(msg="TP"):
            self.assertEqual(self.metric.TP, torch.tensor(1, dtype=torch.int))
        with self.subTest(msg="FN"):
            self.assertEqual(self.metric.FN, torch.tensor(0, dtype=torch.int))
        with self.subTest(msg="FP"):
            self.assertEqual(self.metric.FP, torch.tensor(1, dtype=torch.int))
        with self.subTest(msg="IoU"):
            self.assertEqual(self.metric.IoU, torch.tensor(0.6, dtype=torch.float))
        with self.subTest(msg="DQ"):
            self.assertEqual(dq, torch.tensor(2 / 3, dtype=torch.float))
        with self.subTest(msg="SQ"):
            self.assertEqual(sq, torch.tensor(0.6, dtype=torch.float))
        with self.subTest(msg="PQ"):
            self.assertEqual(pq, torch.tensor(0.6, dtype=torch.float) * torch.tensor(2 / 3, dtype=torch.float))

    def test_5(self):
        pred = (torch.tensor([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                               [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                              [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool),

                torch.tensor([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                              [[0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool))

        target = (torch.tensor([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                                [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                 [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool),

                  torch.tensor([[[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                                 [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                                 [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                                 [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool))

        dq, sq, pq = self.metric(pred, target).values()
        with self.subTest(msg="TP"):
            self.assertEqual(self.metric.TP, torch.tensor(2, dtype=torch.int))
        with self.subTest(msg="FN"):
            self.assertEqual(self.metric.FN, torch.tensor(1, dtype=torch.int))
        with self.subTest(msg="FP"):
            self.assertEqual(self.metric.FP, torch.tensor(2, dtype=torch.int))
        with self.subTest(msg="IoU"):
            self.assertEqual(self.metric.IoU, torch.tensor(1.7, dtype=torch.float))
        with self.subTest(msg="DQ"):
            self.assertEqual(dq, torch.tensor(4 / 7, dtype=torch.float))
        with self.subTest(msg="SQ"):
            self.assertEqual(sq, torch.tensor(0.85, dtype=torch.float))
        with self.subTest(msg="PQ"):
            self.assertEqual(pq, torch.tensor(4 / 7, dtype=torch.float) * torch.tensor(0.85, dtype=torch.float))

    def test_TPs_only(self):
        target = torch.tensor([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool)
        dq, sq, pq = self.metric(target, target).values()
        with self.subTest(msg="TP"):
            self.assertEqual(self.metric.TP, torch.tensor(2, dtype=torch.int))
        with self.subTest(msg="FN"):
            self.assertEqual(self.metric.FN, torch.tensor(0, dtype=torch.int))
        with self.subTest(msg="FP"):
            self.assertEqual(self.metric.FP, torch.tensor(0, dtype=torch.int))
        with self.subTest(msg="IoU"):
            self.assertEqual(self.metric.IoU, torch.tensor(2, dtype=torch.float))
        with self.subTest(msg="DQ"):
            self.assertEqual(dq, torch.tensor(1, dtype=torch.float))
        with self.subTest(msg="SQ"):
            self.assertEqual(sq, torch.tensor(1, dtype=torch.float))
        with self.subTest(msg="PQ"):
            self.assertEqual(pq, torch.tensor(1, dtype=torch.float))

    def test_return_type(self):
        pred = torch.tensor([[[1], [1]], [[0], [1]]], dtype=torch.bool)
        target = torch.tensor([[[1], [0]], [[0], [1]]], dtype=torch.bool)
        dq, sq, pq = self.metric(pred, target).values()
        with self.subTest(msg="TP"):
            self.assertEqual(self.metric.TP, torch.tensor(1, dtype=torch.int))
        with self.subTest(msg="FN"):
            self.assertEqual(self.metric.FN, torch.tensor(1, dtype=torch.int))
        with self.subTest(msg="FP"):
            self.assertEqual(self.metric.FP, torch.tensor(1, dtype=torch.int))
        with self.subTest(msg="IoU"):
            self.assertEqual(self.metric.IoU, torch.tensor(1, dtype=torch.float))
        with self.subTest(msg="DQ"):
            self.assertEqual(dq, torch.tensor(0.5, dtype=torch.float))
        with self.subTest(msg="SQ"):
            self.assertEqual(sq, torch.tensor(1, dtype=torch.float))
        with self.subTest(msg="PQ"):
            self.assertEqual(pq, torch.tensor(0.5, dtype=torch.float))
        with self.subTest(msg="DQ type"):
            self.assertIsInstance(dq, torch.Tensor)
        with self.subTest(msg="SQ type"):
            self.assertIsInstance(sq, torch.Tensor)
        with self.subTest(msg="PQ type"):
            self.assertIsInstance(pq, torch.Tensor)

    def test_empty(self):
        pred = torch.tensor(data=(), dtype=torch.bool)
        target = torch.tensor([[[1], [0]], [[0], [1]]], dtype=torch.bool)
        dq, sq, pq = self.metric(pred, target).values()
        with self.subTest(msg="TP"):
            self.assertEqual(self.metric.TP, torch.tensor(0, dtype=torch.int))
        with self.subTest(msg="FN"):
            self.assertEqual(self.metric.FN, torch.tensor(2, dtype=torch.int))
        with self.subTest(msg="FP"):
            self.assertEqual(self.metric.FP, torch.tensor(0, dtype=torch.int))
        with self.subTest(msg="IoU"):
            self.assertEqual(self.metric.IoU, torch.tensor(0, dtype=torch.float))
        with self.subTest(msg="DQ"):
            self.assertEqual(dq, torch.tensor(0, dtype=torch.float))
        with self.subTest(msg="SQ"):
            self.assertTrue(sq.isnan())
        with self.subTest(msg="PQ"):
            self.assertTrue(pq.isnan())


class TestAJI(unittest.TestCase):
    """Unite test case for the AJI class."""

    def setUp(self) -> None:
        self.metric = AJI()

    def test_0(self):
        pred = torch.tensor([[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                             [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool)

        target = torch.tensor([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool)
        aji = self.metric(pred, target)["AJI"]
        with self.subTest(msg="Intersection"):
            self.assertEqual(self.metric.intersection, torch.tensor(0, dtype=torch.int))
        with self.subTest(msg="Union"):
            self.assertEqual(self.metric.union, torch.tensor(19, dtype=torch.int))
        with self.subTest(msg="AJI"):
            self.assertEqual(aji, torch.tensor(0, dtype=torch.float))

    def test_1(self):
        pred = torch.tensor([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                              [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                              [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                             [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool)

        target = torch.tensor([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool)
        aji = self.metric(pred, target)["AJI"]
        with self.subTest(msg="Intersection"):
            self.assertEqual(self.metric.intersection, torch.tensor(9, dtype=torch.int))
        with self.subTest(msg="Union"):
            self.assertEqual(self.metric.union, torch.tensor(33, dtype=torch.int))
        with self.subTest(msg="AJI"):
            self.assertEqual(aji, torch.tensor(9 / 33, dtype=torch.float))

    def test_2(self):
        pred = torch.tensor([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                             [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool)

        target = torch.tensor([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool)
        aji = self.metric(pred, target)["AJI"]
        with self.subTest(msg="Intersection"):
            self.assertEqual(self.metric.intersection, torch.tensor(11, dtype=torch.int))
        with self.subTest(msg="Union"):
            self.assertEqual(self.metric.union, torch.tensor(19, dtype=torch.int))
        with self.subTest(msg="AJI"):
            self.assertEqual(aji, torch.tensor(11 / 19, dtype=torch.float))

    def test_3(self):
        pred = torch.tensor([[[0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                             [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool)

        target = torch.tensor([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
                                [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool)
        aji = self.metric(pred, target)["AJI"]
        with self.subTest(msg="Intersection"):
            self.assertEqual(self.metric.intersection, torch.tensor(9, dtype=torch.int))
        with self.subTest(msg="Union"):
            self.assertEqual(self.metric.union, torch.tensor(41, dtype=torch.int))
        with self.subTest(msg="AJI"):
            self.assertEqual(aji, torch.tensor(9 / 41, dtype=torch.float))

    def test_4(self):
        pred = torch.tensor([[[0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                             [[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool)

        target = torch.tensor([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool)

        aji = self.metric(pred, target)["AJI"]
        with self.subTest(msg="Intersection"):
            self.assertEqual(self.metric.intersection, torch.tensor(6, dtype=torch.int))
        with self.subTest(msg="Union"):
            self.assertEqual(self.metric.union, torch.tensor(14, dtype=torch.int))
        with self.subTest(msg="AJI"):
            self.assertEqual(aji, torch.tensor(6 / 14, dtype=torch.float))

    def test_5(self):
        pred = (torch.tensor([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                               [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                              [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool),

                torch.tensor([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                              [[0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool))

        target = (torch.tensor([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                                [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                 [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool),

                  torch.tensor([[[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                                 [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                                 [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                                 [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool))

        aji = self.metric(pred, target)["AJI"]
        with self.subTest(msg="Intersection"):
            self.assertEqual(self.metric.intersection, torch.tensor(29, dtype=torch.int))
        with self.subTest(msg="Union"):
            self.assertEqual(self.metric.union, torch.tensor(49, dtype=torch.int))
        with self.subTest(msg="AJI"):
            self.assertEqual(aji, torch.tensor(29 / 49, dtype=torch.float))

    def test_1_pred_shuffled(self):
        pred = torch.tensor([[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                             [[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                              [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                              [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool)

        target = torch.tensor([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool)
        aji = self.metric(pred, target)["AJI"]
        with self.subTest(msg="Intersection"):
            self.assertEqual(self.metric.intersection, torch.tensor(9, dtype=torch.int))
        with self.subTest(msg="Union"):
            self.assertEqual(self.metric.union, torch.tensor(24, dtype=torch.int))
        with self.subTest(msg="AJI"):
            self.assertEqual(aji, torch.tensor(9 / 24, dtype=torch.float))

    def test_TPs_only(self):
        target = torch.tensor([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool)

        aji = self.metric(target, target)["AJI"]
        with self.subTest(msg="Intersection"):
            self.assertEqual(self.metric.intersection, torch.tensor(19, dtype=torch.int))
        with self.subTest(msg="Union"):
            self.assertEqual(self.metric.union, torch.tensor(19, dtype=torch.int))
        with self.subTest(msg="AJI"):
            self.assertEqual(aji, torch.tensor(1, dtype=torch.float))

    def test_return_type(self):
        pred = torch.tensor([[[1], [1]], [[0], [1]]], dtype=torch.bool)
        target = torch.tensor([[[1], [0]], [[0], [1]]], dtype=torch.bool)
        aji = self.metric(pred, target)["AJI"]
        with self.subTest(msg="Intersection"):
            self.assertEqual(self.metric.intersection, torch.tensor(2, dtype=torch.int))
        with self.subTest(msg="Union"):
            self.assertEqual(self.metric.union, torch.tensor(3, dtype=torch.int))
        with self.subTest(msg="AJI"):
            self.assertEqual(aji, torch.tensor(2 / 3, dtype=torch.float))
        with self.subTest(msg="AJI type"):
            self.assertIsInstance(aji, torch.Tensor)

    def test_empty(self):
        pred = torch.tensor(data=(), dtype=torch.bool)
        target = torch.tensor([[[1], [0]], [[0], [1]]], dtype=torch.bool)
        aji = self.metric(pred, target)["AJI"]
        with self.subTest(msg="Intersection"):
            self.assertEqual(self.metric.intersection, torch.tensor(0, dtype=torch.int))
        with self.subTest(msg="Union"):
            self.assertEqual(self.metric.union, torch.tensor(2, dtype=torch.int))
        with self.subTest(msg="AJI"):
            self.assertEqual(aji, torch.tensor(0, dtype=torch.float))


class TestModAJI(unittest.TestCase):
    """Unite test case for the ModAJI class."""

    def setUp(self) -> None:
        self.metric = ModAJI()

    def test_0(self):
        pred = torch.tensor([[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                             [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool)

        target = torch.tensor([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool)
        aji = self.metric(pred, target)["AJI"]
        with self.subTest(msg="Intersection"):
            self.assertEqual(self.metric.intersection, torch.tensor(0, dtype=torch.int))
        with self.subTest(msg="Union"):
            self.assertEqual(self.metric.union, torch.tensor(19, dtype=torch.int))
        with self.subTest(msg="AJI"):
            self.assertEqual(aji, torch.tensor(0, dtype=torch.float))

    def test_1(self):
        pred = torch.tensor([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                              [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                              [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                             [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool)

        target = torch.tensor([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool)
        aji = self.metric(pred, target)["AJI"]
        with self.subTest(msg="Intersection"):
            self.assertEqual(self.metric.intersection, torch.tensor(9, dtype=torch.int))
        with self.subTest(msg="Union"):
            self.assertEqual(self.metric.union, torch.tensor(24, dtype=torch.int))
        with self.subTest(msg="AJI"):
            self.assertEqual(aji, torch.tensor(9 / 24, dtype=torch.float))

    def test_2(self):
        pred = torch.tensor([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                             [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool)

        target = torch.tensor([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool)
        aji = self.metric(pred, target)["AJI"]
        with self.subTest(msg="Intersection"):
            self.assertEqual(self.metric.intersection, torch.tensor(11, dtype=torch.int))
        with self.subTest(msg="Union"):
            self.assertEqual(self.metric.union, torch.tensor(19, dtype=torch.int))
        with self.subTest(msg="AJI"):
            self.assertEqual(aji, torch.tensor(11 / 19, dtype=torch.float))

    def test_3(self):
        pred = torch.tensor([[[0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                             [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool)

        target = torch.tensor([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
                                [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool)
        aji = self.metric(pred, target)["AJI"]
        with self.subTest(msg="Intersection"):
            self.assertEqual(self.metric.intersection, torch.tensor(9, dtype=torch.int))
        with self.subTest(msg="Union"):
            self.assertEqual(self.metric.union, torch.tensor(35, dtype=torch.int))
        with self.subTest(msg="AJI"):
            self.assertEqual(aji, torch.tensor(9 / 35, dtype=torch.float))

    def test_4(self):
        pred = torch.tensor([[[0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                             [[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool)

        target = torch.tensor([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool)

        aji = self.metric(pred, target)["AJI"]
        with self.subTest(msg="Intersection"):
            self.assertEqual(self.metric.intersection, torch.tensor(6, dtype=torch.int))
        with self.subTest(msg="Union"):
            self.assertEqual(self.metric.union, torch.tensor(14, dtype=torch.int))
        with self.subTest(msg="AJI"):
            self.assertEqual(aji, torch.tensor(6 / 14, dtype=torch.float))

    def test_5(self):
        pred = (torch.tensor([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                               [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                              [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool),

                torch.tensor([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                              [[0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool))

        target = (torch.tensor([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                                [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                 [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool),

                  torch.tensor([[[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                                 [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                                 [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                                 [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool))

        aji = self.metric(pred, target)["AJI"]
        with self.subTest(msg="Intersection"):
            self.assertEqual(self.metric.intersection, torch.tensor(29, dtype=torch.int))
        with self.subTest(msg="Union"):
            self.assertEqual(self.metric.union, torch.tensor(49, dtype=torch.int))
        with self.subTest(msg="AJI"):
            self.assertEqual(aji, torch.tensor(29 / 49, dtype=torch.float))

    def test_1_pred_shuffled(self):
        pred = torch.tensor([[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                             [[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                              [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                              [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool)

        target = torch.tensor([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool)
        aji = self.metric(pred, target)["AJI"]
        with self.subTest(msg="Intersection"):
            self.assertEqual(self.metric.intersection, torch.tensor(9, dtype=torch.int))
        with self.subTest(msg="Union"):
            self.assertEqual(self.metric.union, torch.tensor(24, dtype=torch.int))
        with self.subTest(msg="AJI"):
            self.assertEqual(aji, torch.tensor(9 / 24, dtype=torch.float))

    def test_TPs_only(self):
        target = torch.tensor([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool)

        aji = self.metric(target, target)["AJI"]
        with self.subTest(msg="Intersection"):
            self.assertEqual(self.metric.intersection, torch.tensor(19, dtype=torch.int))
        with self.subTest(msg="Union"):
            self.assertEqual(self.metric.union, torch.tensor(19, dtype=torch.int))
        with self.subTest(msg="AJI"):
            self.assertEqual(aji, torch.tensor(1, dtype=torch.float))

    def test_return_type(self):
        pred = torch.tensor([[[1], [1]], [[0], [1]]], dtype=torch.bool)
        target = torch.tensor([[[1], [0]], [[0], [1]]], dtype=torch.bool)
        aji = self.metric(pred, target)["AJI"]
        with self.subTest(msg="Intersection"):
            self.assertEqual(self.metric.intersection, torch.tensor(2, dtype=torch.int))
        with self.subTest(msg="Union"):
            self.assertEqual(self.metric.union, torch.tensor(3, dtype=torch.int))
        with self.subTest(msg="AJI"):
            self.assertEqual(aji, torch.tensor(2 / 3, dtype=torch.float))
        with self.subTest(msg="AJI type"):
            self.assertIsInstance(aji, torch.Tensor)

    def test_empty(self):
        pred = torch.tensor(data=(), dtype=torch.bool)
        target = torch.tensor([[[1], [0]], [[0], [1]]], dtype=torch.bool)
        aji = self.metric(pred, target)["AJI"]
        with self.subTest(msg="Intersection"):
            self.assertEqual(self.metric.intersection, torch.tensor(0, dtype=torch.int))
        with self.subTest(msg="Union"):
            self.assertEqual(self.metric.union, torch.tensor(2, dtype=torch.int))
        with self.subTest(msg="AJI"):
            self.assertEqual(aji, torch.tensor(0, dtype=torch.float))

if __name__ == '__main__':
    unittest.main()
