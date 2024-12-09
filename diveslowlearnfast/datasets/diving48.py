import json
import math
import random

import av
import os
import time

import numpy as np

from torch.utils.data import Dataset


def decord_load_video(video_path, num_frames, temporal_random_jitter=0, temporal_random_offset=0, **kwargs):
    import decord
    from decord import cpu

    vr = decord.VideoReader(video_path, ctx=cpu(0))
    num_frames = num_frames if num_frames <= len(vr) else len(vr)
    indices = np.linspace(0, len(vr) - 1, num_frames, dtype=np.uint32)
    indices = temporal_random_offset_indices(indices, len(vr) - 1, temporal_random_offset)
    indices = temporal_random_jitter_indices(indices, len(vr) - 1, num_frames, temporal_random_jitter)
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

    jitter = np.random.randint(-temporal_random_jitter, temporal_random_jitter + 1, size=num_frames)
    clipped = np.clip(indices + jitter, 0, total_frames)
    return np.sort(clipped)


def load_video_av_optimized(video_path, num_frames, multi_thread_decode=False, temporal_random_jitter=0, temporal_random_offset=0, **kwargs):
    """Efficiently load video frames using uniform sampling"""
    container = av.open(video_path)
    if multi_thread_decode:
        container.streams.video[0].thread_type = 'AUTO'
    video_stream = container.streams.video[0]
    total_frames = video_stream.frames
    num_frames = total_frames if num_frames == -1 else num_frames

    # Calculate timestamps for uniform sampling
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=np.uint32)
    indices = temporal_random_offset_indices(indices, total_frames - 1, temporal_random_offset)
    indices = temporal_random_jitter_indices(indices, total_frames - 1, num_frames, temporal_random_jitter)
    frames = []

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

    def __init__(self,
                 dataset_path,
                 num_frames,
                 dataset_type='train',
                 dataset_version='V2',
                 temporal_random_jitter=0,
                 temporal_random_offset=0,
                 transform_fn=None,
                 target_fps=None,
                 use_decord=False,
                 multi_thread_decode=False):
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
        self.multi_thread_decode = multi_thread_decode
        self._init_dataset()

    def _init_dataset(self):
        with open(self.annotations_path, 'rb') as f:
            self.data = json.loads(f.read())

        with open(self.vocab_path, 'rb') as f:
            self.vocab = json.loads(f.read())

    def _read_frames(self, video_id):
        start = time.time()
        video_path = os.path.join(self.videos_path, f'{video_id}.mp4')

        frames = self.load_video(video_path,
                                 self.num_frames,
                                 self.temporal_random_jitter,
                                 self.temporal_random_offset,
                                 multithread_decode=self.multi_thread_decode)
        io_time = time.time() - start

        if len(frames) < self.num_frames:
            frames = pad_video(frames, self.num_frames)

        start = time.time()
        frames = self._transform(frames)
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

    def _transform(self, frames):
        if self.transform_fn:
            frames = self.transform_fn(frames)

        return frames
