import json
import math
import random

import av
import os
import time
import torch

import numpy as np

from pytorchvideo.transforms import ShortSideScale, Div255, UniformTemporalSubsample
from pytorchvideo.transforms.functional import uniform_crop
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from torchvision.transforms.v2 import Lambda

def decord_load_video(video_path, num_frames, temporal_random_jitter=0, temporal_random_offset=0):
    import decord
    from decord import cpu

    vr = decord.VideoReader(video_path, ctx=cpu(0))
    num_frames = num_frames if num_frames <= len(vr) else len(vr)
    indices = np.linspace(0, len(vr)-1, num_frames, dtype=np.uint32)
    indices = temporal_random_offset_indices(indices, len(vr)-1, temporal_random_offset)
    indices = temporal_random_jitter_indices(indices, len(vr)-1, num_frames, temporal_random_jitter)
    frames = vr.get_batch(indices)
    return frames.asnumpy()


def pad_video(video, size):
    padding_size = size - len(video)
    return np.concatenate((video, np.zeros((padding_size, *video.shape[1:]))))

def temporal_random_offset_indices(indices, total_frames, temporal_random_offset=0):
    if temporal_random_offset == 0:
        return indices

    return np.clip(math.floor(random.uniform(0, temporal_random_offset)) + indices, 0, total_frames)

def temporal_random_jitter_indices(indices, total_frames, num_frames, temporal_random_jitter=0):
    if temporal_random_jitter == 0:
        return indices

    jitter = np.random.randint(-temporal_random_jitter, temporal_random_jitter+1, size=num_frames)
    clipped = np.clip(indices + jitter, 0, total_frames)
    return np.sort(clipped)

def load_video_av_optimized(video_path, num_frames, temporal_random_jitter=0, temporal_random_offset=0):
    """Efficiently load video frames using uniform sampling"""
    container = av.open(video_path)
    video_stream = container.streams.video[0]
    total_frames = video_stream.frames
    num_frames = total_frames if num_frames == -1 else num_frames

    # Calculate timestamps for uniform sampling
    indices = np.linspace(0, total_frames-1, num_frames, dtype=np.uint32)
    indices = temporal_random_offset_indices(indices, total_frames-1, temporal_random_offset)
    indices = temporal_random_jitter_indices(indices, total_frames-1, num_frames, temporal_random_jitter)
    frames = []
    print(indices, len(indices))

    for idx, frame in enumerate(container.decode(video=0)):
        counts = len(indices[np.isin(indices, idx)])
        if counts > 0:
            frames.extend(counts * [frame.to_ndarray(format='rgb24')])

    container.close()
    return np.stack(frames)

def collate_fn(batch):
    max_frames = max(video.shape[0] for video in batch)
    return [pad_video(video, max_frames) for video in batch]

class Diving48Dataset(Dataset):

    def __init__(self, dataset_path, num_frames,
                 dataset_type='train',
                 dataset_version='V2',
                 temporal_random_jitter=0,
                 temporal_random_offset=0,
                 transform_fn=None,
                 target_fps=None,
                 use_decord=False):
        super().__init__()
        assert dataset_type in ['train', 'test']
        self.videos_path = os.path.join(dataset_path, 'rgb')
        self.annotations_path = os.path.join(dataset_path, f'Diving48_{dataset_version}_{dataset_type}.json')
        self.vocab_path = os.path.join(dataset_path, f'Diving48_vocab.json')
        self.num_frames = num_frames
        self.target_fps = target_fps
        self.transform_fn = transform_fn
        self.temporal_random_jitter = temporal_random_jitter
        self.temporal_random_offset = temporal_random_offset
        self.load_video = decord_load_video if use_decord else load_video_av_optimized
        self._init_dataset()

    def _init_dataset(self):
        with open(self.annotations_path, 'rb') as f:
            self.data = json.loads(f.read())

        with open(self.vocab_path, 'rb') as f:
            self.vocab = json.loads(f.read())

    def _read_frames(self, video_id):
        start = time.time()
        video_path = os.path.join(self.videos_path, f'{video_id}.mp4')

        frames = self.load_video(video_path, self.num_frames, self.temporal_random_jitter, self.temporal_random_offset)
        io_time = time.time() - start

        if len(frames) < self.num_frames:
            frames = pad_video(frames, self.num_frames)

        transform_time = None
        if self.transform_fn:
            start = time.time()
            frames = self.transform_fn(frames)
            transform_time = time.time() - start

        return frames, io_time, transform_time

    def __getitem__(self, index):
        video_id = self.data[index]['vid_name']
        label = self.data[index]['label']
        frames, io_time, transform_time = self._read_frames(video_id)
        return frames, label, io_time, transform_time


    def __len__(self):
        return self.num_videos

    @property
    def num_videos(self):
        return len(self.data)

    def get_label(self, idx):
        return self.vocab[idx]


import unittest

DATASET_PATH = os.environ.get('DATASET_PATH')

class Diving48DatasetTest(unittest.TestCase):

    def test_init(self):
        diving48 = Diving48Dataset(DATASET_PATH, num_frames=4)
        self.assertEqual(diving48.videos_path, os.path.join(DATASET_PATH, 'rgb'))
        self.assertEqual(diving48.annotations_path, os.path.join(DATASET_PATH, 'Diving48_V2_train.json'))
        self.assertEqual(diving48.target_fps, None)

    def test_get_item(self):
        diving48 = Diving48Dataset(DATASET_PATH,num_frames=4)
        diving48_iter = iter(diving48)
        video, label, *_ = next(diving48_iter)
        self.assertIsNotNone(label)
        self.assertGreater(len(video), 0)


    def test_transforms(self):
        transform = Compose([
            Lambda(lambda x: torch.tensor(x).permute(3, 0, 1, 2)),
            Div255(), # Div255 assumes C x T x W x H
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
            Div255(), # Div255 assumes C x T x W x H
            ShortSideScale(size=256),
            Lambda(lambda x: uniform_crop(x, size=224, spatial_idx=1)),
            UniformTemporalSubsample(128) # ensures each sample has the same number of frames, will under sample if T < 128, and over sample if T > 128
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


if __name__ == '__main__':
    unittest.main()