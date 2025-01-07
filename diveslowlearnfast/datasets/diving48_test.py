import os
import unittest

import numpy as np
import torch
from pytorchvideo.transforms import ShortSideScale, Div255, UniformTemporalSubsample
from pytorchvideo.transforms.functional import uniform_crop
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Lambda, Compose

from diveslowlearnfast.config import Config
from diveslowlearnfast.datasets import Diving48Dataset
from diveslowlearnfast.datasets.diving48 import pad_video, collate_fn
from diveslowlearnfast.train import MultigridSchedule

DATASET_PATH = os.environ.get('DATASET_PATH')


class Diving48DatasetTest(unittest.TestCase):

    def test_init(self):
        diving48 = Diving48Dataset(DATASET_PATH, num_frames=4)
        self.assertEqual(diving48.videos_path, os.path.join(DATASET_PATH, 'rgb'))
        self.assertEqual(diving48.annotations_path, os.path.join(DATASET_PATH, 'Diving48_V2_train.json'))
        self.assertEqual(diving48.target_fps, None)

    def test_get_item(self):
        diving48 = Diving48Dataset(DATASET_PATH, num_frames=4)
        diving48_iter = iter(diving48)
        video, label, *_ = next(diving48_iter)
        self.assertIsNotNone(label)
        self.assertGreater(len(video), 0)

    def test_transforms(self):
        transform = Compose([
            Lambda(lambda x: torch.tensor(x).permute(3, 0, 1, 2)),
            Div255(),  # Div255 assumes C x T x W x H
            ShortSideScale(size=256),
            Lambda(lambda x: uniform_crop(x, size=224, spatial_idx=1)),
        ])
        diving48 = Diving48Dataset(DATASET_PATH, num_frames=4, transform_fn=transform)
        x, *_ = next(iter(diving48))
        # assert dims equal to 3 x T x 224 x 224
        self.assertEqual(x.shape[0], 3)
        self.assertEqual(x.shape[2:], (224, 224))

    def test_dataloader(self):
        transform = Compose([
            Lambda(lambda x: torch.tensor(x).permute(3, 0, 1, 2)),
            Div255(),  # Div255 assumes C x T x W x H
            ShortSideScale(size=256),
            Lambda(lambda x: uniform_crop(x, size=224, spatial_idx=1)),
            UniformTemporalSubsample(128)
            # ensures each sample has the same number of frames, will under sample if T < 128, and over sample if T > 128
        ])
        diving48 = Diving48Dataset(DATASET_PATH, num_frames=2, transform_fn=transform)
        diving48_dataloader = DataLoader(diving48, batch_size=4, shuffle=True)
        diving48_iter = iter(diving48_dataloader)
        x, y, *_ = next(diving48_iter)
        self.assertEqual(y.size()[0], 4)
        # assert shape B x C x T x W x H
        self.assertEqual(x.size(), (4, 3, 128, 224, 224))

    def test_pad_video(self):
        x = np.random.rand(10, 24, 24, 3)
        x_padded = pad_video(x, 12)
        self.assertEqual(x_padded.shape, (12, 24, 24, 3))
        zero = np.sum(x_padded[10:], axis=(0, 1, 2, 3))
        self.assertEqual(zero, 0)
        not_zero = np.sum(x_padded[:10], axis=(0, 1, 2, 3))
        self.assertNotEqual(not_zero, 0)

    def test_collate_fn(self):
        xb = [
            np.random.rand(12, 24, 24, 3),
            np.random.rand(64, 24, 24, 3),
            np.random.rand(8, 24, 24, 3),
            np.random.rand(128, 24, 24, 3)
        ]
        xb_padded = collate_fn(xb)
        xb_padded = np.stack(xb_padded)
        self.assertEqual(xb_padded.shape, (4, 128, 24, 24, 3))

    def test_get_label(self):
        diving48 = Diving48Dataset(DATASET_PATH, num_frames=4)
        self.assertIsNotNone(diving48.get_label(0))
        self.assertIsNotNone(diving48.get_label(47))
        self.assertIsNotNone(len(diving48.vocab), 48)

    def test_multigrid(self):
        cfg = Config()
        multigrid_schedule = MultigridSchedule()
        cfg.MULTIGRID.LONG_CYCLE = True
        cfg.MULTIGRID.SHORT_CYCLE = True
        cfg.SOLVER.STEPS = [0, 16, 24, 32]
        cfg.SOLVER.MAX_EPOCH = 32
        cfg.TRAIN.BATCH_SIZE = 16
        cfg = multigrid_schedule.init_multigrid(cfg)
        cfg, _ = multigrid_schedule.update_long_cycle(cfg, 0)
        diving48 = Diving48Dataset(DATASET_PATH,
                                   dataset_type='test',
                                   num_frames=16)
        multigrid_schedule.set_dataset(diving48, cfg)
        dataset = iter(diving48)
        x, *_ = next(dataset)
        self.assertEqual(x.size(), torch.Size((3, 8, 112, 112)))
        multigrid_schedule.step(cfg)
        x, *_ = next(dataset)
        self.assertEqual(x.size(), torch.Size((3, 8, 158, 158)))
        multigrid_schedule.step(cfg)
        x, *_ = next(dataset)
        self.assertEqual(x.size(), torch.Size((3, 8, 158, 158)))
        multigrid_schedule.step(cfg)
        x, *_ = next(dataset)
        self.assertEqual(x.size(), torch.Size((3, 8, 112, 112)))

    def test_threshold(self):
        diving48 = Diving48Dataset(DATASET_PATH,
                                   dataset_type='train',
                                   num_frames=2,
                                   threshold=800)
        # for this threshold we will only have 4 classes remaining
        self.assertEqual(diving48.num_videos, 3200)
        self.assertEqual(diving48.num_classes, 4)


if __name__ == '__main__':
    unittest.main()
