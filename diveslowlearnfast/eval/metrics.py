import unittest

import numpy as np

def iou(gt, pred, dim=(0, 1)):
    """
    Calculate the Intersection over Union (IoU) between two sets of predictions.
    Args:
        gt: np.ndarray binary ground truth matrix or matrices
        pred: np.ndarray binary ground truth matrix or matrices
        dim: the dimension over which to compute the IoU
    Returns:
        iou: float between 0 and 1
    """
    overlap = gt * pred
    union = (gt + pred) > 0
    return overlap.sum(axis=dim) / union.sum(axis=dim)


def wr(gt, exp, dim=(0, 1)):
    """
    Calculates the wrong-reason score between the explanation and ground truth explanation.
    bin(expl(X) * M).sum() / M.sum() where bin is the binarysation, expl is the explanation for an input X, and M
    is the ground truth explanation.
    Args:
        gt: np.ndarray binary ground truth matrix or matrices
        exp: np.ndarray binary explanation matrix or matrices
    Returns:
        wr: float between 0 and 1
    """
    overlap = gt * exp
    return overlap.sum(axis=dim) / exp.sum(axis=dim)


class MetricsTest(unittest.TestCase):


    def test_iou(self):
        target_mask = np.zeros((8, 8))
        target_mask[1:5, 1:5] = 1

        pred_mask = np.zeros((8, 8))
        pred_mask[3:7, 3:7] = 1

        actual = 16 / (128 - 16)
        score = iou(target_mask, pred_mask)
        self.assertEqual(score, actual)
        score = iou(target_mask, target_mask)
        self.assertEqual(score, 1.0)

    def test_iou_batch(self):
        target_mask = np.zeros((8, 8))
        target_mask[1:5, 1:5] = 1
        target_masks = np.stack([target_mask, target_mask])

        pred_mask = np.zeros((8, 8))
        pred_mask[3:7, 3:7] = 1
        pred_masks = np.stack([pred_mask, pred_mask])
        actual = 16 / (128 - 16)
        scores = iou(target_masks, pred_masks, dim=(1, 2))
        self.assertEqual(scores.tolist(), [actual, actual])




