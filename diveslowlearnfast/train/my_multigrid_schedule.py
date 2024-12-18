import unittest

from diveslowlearnfast.config import Config


class MyMultiGridSchedule:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._init_multigrid()

    def _init_multigrid(self):
        shapes = self.cfg.MULTIGRID.LONG_CYCLE_FACTORS
        B = self.cfg.TRAIN.BATCH_SIZE
        T = self.cfg.DATA.NUM_FRAMES
        HW = self.cfg.DATA.TRAIN_CROP_SIZE

        t = shapes[0][0]
        hw = int(HW*shapes[0][1])
        b = int(B*(T*t)*(HW/hw)**2)

        self.cfg.TRAIN.BATCH_SIZE = b
        self.cfg.DATA.NUM_FRAMES = int(T*t)
        self.cfg.DATA.TRAIN_CROP_SIZE = hw

        max_len = len(self.cfg.MULTIGRID.LONG_CYCLE_FACTORS)-1
        idx = 0
        self.schedule = []
        for epoch in self.cfg.SOLVER.STEPS:
            self.schedule.append((idx, epoch))
            if idx+1 > max_len:
                idx = 0
            else:
                idx += 1

    def _current_long_cycle_factors(self, cfg, epoch):
        for idx, step in self.schedule:
            if epoch >= step:
                return cfg.MULTIGRID.LONG_CYCLE_FACTORS[idx]

        # fallback to last cycle
        return cfg.MULTIGRID.LONG_CYCLE_FACTORS[-1]


class MyMultiGridScheduleTest(unittest.TestCase):

    def test_init_multigrid(self):
        cfg = Config()
        cfg.SOLVER.STEPS = [0, 16, 24, 32, 36]
        self.assertEqual(cfg.TRAIN.BATCH_SIZE, 4)
        self.assertEqual(cfg.DATA.NUM_FRAMES, 32)
        self.assertEqual(cfg.DATA.TRAIN_CROP_SIZE, 224)

        mg = MyMultiGridSchedule(cfg)
        self.assertEqual(mg.cfg.TRAIN.BATCH_SIZE, 64)
        self.assertEqual(mg.cfg.DATA.NUM_FRAMES, 8)
        self.assertEqual(mg.cfg.DATA.TRAIN_CROP_SIZE, 158)
        self.assertEqual(mg.schedule[0], (0, 0))
        self.assertEqual(mg.schedule[1], (1, 16))
        self.assertEqual(mg.schedule[2], (2, 24))
        self.assertEqual(mg.schedule[3], (3, 32))
        self.assertEqual(mg.schedule[4], (0, 36))
