import unittest

import torch

from diveslowlearnfast.config import Config
from diveslowlearnfast.datasets import Diving48Dataset
from diveslowlearnfast.models import SlowFast
from diveslowlearnfast.train import MultigridSchedule
from diveslowlearnfast.train.helper import get_train_objects

DATASET_PATH = '/home/s2871513/Datasets/Diving48'

class SlowFastTest(unittest.TestCase):


    def test_multigrid(self):
        cfg = Config()
        cfg.MULTIGRID.LONG_CYCLE = True
        cfg.MULTIGRID.SHORT_CYCLE = True
        cfg.SOLVER.STEPS = [0, 16, 24, 32]
        cfg.SOLVER.MAX_EPOCH = 32
        cfg.TRAIN.BATCH_SIZE = 16
        cfg.DATA.DATASET_PATH = DATASET_PATH

        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        cfg, _ = multigrid.update_long_cycle(cfg, 0)
        model = SlowFast(cfg)
        _, _, loader, diving48, scaler = get_train_objects(cfg, model)
        multigrid.set_dataset(diving48, cfg)
        xb, yb = next(iter(loader))
        self.assertEqual(xb.size(), torch.Size((32, 3, 8, 158, 158)))


if __name__ == '__main__':
    unittest.main()
