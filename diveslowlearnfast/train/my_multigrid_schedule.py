import unittest

from diveslowlearnfast.config import Config


class MyMultiGridSchedule:

    def __init__(self, cfg: Config):
        self._B = cfg.TRAIN.BATCH_SIZE
        self._T = cfg.DATA.NUM_FRAMES
        self._HW = cfg.DATA.TRAIN_CROP_SIZE
        self._init_multigrid(cfg)

    def _init_multigrid(self, cfg: Config):
        max_len = len(cfg.MULTIGRID.LONG_CYCLE_FACTORS) - 1
        idx = 0
        self.schedule = []
        for epoch in cfg.SOLVER.STEPS:
            self.schedule.append((idx, epoch))
            if idx + 1 > max_len:
                idx = 0
            else:
                idx += 1

    def get_current_long_cycle(self, cfg, epoch):
        shapes = self._get_current_long_cycle_factors(cfg, epoch)
        B = self._B
        T = self._T
        HW = self._HW

        hw_factor = shapes[1]
        if cfg.MULTIGRID.SHORT_CYCLE:
            short_cycle_idx = epoch % cfg.MULTIGRID.SHORT_CYCLE_PERIOD
            if short_cycle_idx in [0, 1]:
                hw_factor = cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]

        t = int(T * shapes[0])
        hw = int(HW * hw_factor)
        b = int(B * (T // t) * (HW / hw) ** 2)

        cfg.TRAIN.BATCH_SIZE = b
        cfg.DATA.NUM_FRAMES = t
        cfg.DATA.TRAIN_CROP_SIZE = hw

        return cfg

    def _get_current_long_cycle_factors(self, cfg, epoch):
        for idx, step in self.schedule:
            if step >= epoch:
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
        self.assertEqual(mg._B, 4)
        self.assertEqual(mg._T, 32)
        self.assertEqual(mg._HW, 224)
        self.assertEqual(mg.schedule[0], (0, 0))
        self.assertEqual(mg.schedule[1], (1, 16))
        self.assertEqual(mg.schedule[2], (2, 24))
        self.assertEqual(mg.schedule[3], (3, 32))
        self.assertEqual(mg.schedule[4], (0, 36))

    def test_get_current_long_cycle_factors(self):
        cfg = Config()
        cfg.SOLVER.STEPS = [0, 16, 24, 32, 36]
        mg = MyMultiGridSchedule(cfg)
        shape = mg._get_current_long_cycle_factors(cfg, 0)
        self.assertEqual(shape, (0.25, 0.5 ** 0.5))

        shape = mg._get_current_long_cycle_factors(cfg, 16)
        self.assertEqual(shape, (0.5, 0.5 ** 0.5))

        shape = mg._get_current_long_cycle_factors(cfg, 24)
        self.assertEqual(shape, (0.5, 1))

        shape = mg._get_current_long_cycle_factors(cfg, 32)
        self.assertEqual(shape, (1, 1))

        shape = mg._get_current_long_cycle_factors(cfg, 36)
        self.assertEqual(shape, (0.25, 0.5 ** 0.5))

    def test_get_current_long_cycle(self):
        cfg = Config()
        cfg.SOLVER.STEPS = [0, 16, 24, 32, 36]
        self.assertEqual(cfg.TRAIN.BATCH_SIZE, 4)
        self.assertEqual(cfg.DATA.NUM_FRAMES, 32)
        self.assertEqual(cfg.DATA.TRAIN_CROP_SIZE, 224)

        mg = MyMultiGridSchedule(cfg)
        cfg = mg.get_current_long_cycle(cfg, 0)
        self.assertEqual(cfg.TRAIN.BATCH_SIZE, 32)
        self.assertEqual(cfg.DATA.NUM_FRAMES, 8)
        self.assertEqual(cfg.DATA.TRAIN_CROP_SIZE, 158)

        cfg = mg.get_current_long_cycle(cfg, 16)
        self.assertEqual(cfg.TRAIN.BATCH_SIZE, 16)
        self.assertEqual(cfg.DATA.NUM_FRAMES, 16)
        self.assertEqual(cfg.DATA.TRAIN_CROP_SIZE, 158)

        cfg = mg.get_current_long_cycle(cfg, 24)
        self.assertEqual(cfg.TRAIN.BATCH_SIZE, 8)
        self.assertEqual(cfg.DATA.NUM_FRAMES, 16)
        self.assertEqual(cfg.DATA.TRAIN_CROP_SIZE, 224)

        cfg = mg.get_current_long_cycle(cfg, 32)
        self.assertEqual(cfg.TRAIN.BATCH_SIZE, 4)
        self.assertEqual(cfg.DATA.NUM_FRAMES, 32)
        self.assertEqual(cfg.DATA.TRAIN_CROP_SIZE, 224)

        cfg = mg.get_current_long_cycle(cfg, 0)
        self.assertEqual(cfg.TRAIN.BATCH_SIZE, 32)
        self.assertEqual(cfg.DATA.NUM_FRAMES, 8)
        self.assertEqual(cfg.DATA.TRAIN_CROP_SIZE, 158)


    def test_get_current_long_cycle_short_cycle(self):
        cfg = Config()
        cfg.SOLVER.STEPS = [0, 16, 24, 32, 36]
        cfg.MULTIGRID.SHORT_CYCLE = True
        cfg.MULTIGRID.SHORT_CYCLE_PERIOD = 3

        mg = MyMultiGridSchedule(cfg)
        cfg = mg.get_current_long_cycle(cfg, 0)
        self.assertEqual(cfg.TRAIN.BATCH_SIZE, 64)
        self.assertEqual(cfg.DATA.NUM_FRAMES, 8)
        self.assertEqual(cfg.DATA.TRAIN_CROP_SIZE, 112)

        cfg = mg.get_current_long_cycle(cfg, 1)
        self.assertEqual(cfg.TRAIN.BATCH_SIZE, 32)
        self.assertEqual(cfg.DATA.NUM_FRAMES, 8)
        self.assertEqual(cfg.DATA.TRAIN_CROP_SIZE, 158)

        cfg = mg.get_current_long_cycle(cfg, 2)
        self.assertEqual(cfg.TRAIN.BATCH_SIZE, 32)
        self.assertEqual(cfg.DATA.NUM_FRAMES, 8)
        self.assertEqual(cfg.DATA.TRAIN_CROP_SIZE, 158)

        cfg = mg.get_current_long_cycle(cfg, 3)
        self.assertEqual(cfg.TRAIN.BATCH_SIZE, 64)
        self.assertEqual(cfg.DATA.NUM_FRAMES, 8)
        self.assertEqual(cfg.DATA.TRAIN_CROP_SIZE, 112)

