import json
import math
import random

import av
import os
import time
import logging

import numpy as np

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'ERROR'))


def decord_load_video(video_path, num_frames, temporal_random_jitter=0, temporal_random_offset=0,
                      use_sampling_ratio=False, **kwargs):
    import decord
    from decord import cpu

    vr = decord.VideoReader(video_path, ctx=cpu(0))
    num_frames = num_frames if num_frames <= len(vr) else len(vr)
    indices = np.linspace(0, len(vr) - 1, num_frames, dtype=np.int32)
    indices = temporal_random_offset_indices(indices, len(vr) - 1, temporal_random_offset, use_sampling_ratio)
    indices = temporal_random_jitter_indices(indices, len(vr) - 1, num_frames, temporal_random_jitter)
    frames = vr.get_batch(indices)
    return frames.asnumpy()


def pad_video(video, size):
    padding_size = size - len(video)
    return np.concatenate((video, np.zeros((padding_size, *video.shape[1:]))))


def wrap_around(offset_indices, total_frames):
    result = []
    for idx in offset_indices:
        if idx > total_frames:
            diff = abs(offset_indices[-1] - offset_indices[-2])
            result.insert(0, result[0] - diff)
        else:
            result.append(idx)

    return np.clip(result, 0, total_frames)


def temporal_random_offset_indices(indices, total_frames, temporal_random_offset=0, use_sampling_ratio=False):
    if temporal_random_offset == 0 and not use_sampling_ratio:
        return indices

    if use_sampling_ratio:
        # ratio between total number frames and number of frames being sampled
        # total_frames/num_frames is the minimum by which we need to shift the indices in order to cover all frames
        # across different epochs. Calculating this dynamically may result in more stable behaviour.
        temporal_random_offset = math.floor(total_frames / len(indices))
        logger.debug(
            f"use_sampling_ratio = True, calculating temporal_random_offset: {total_frames}/{len(indices)}={temporal_random_offset}")

    offset_indices = math.floor(random.uniform(0, temporal_random_offset)) + indices
    return wrap_around(offset_indices, total_frames)


def temporal_random_jitter_indices(indices, total_frames, num_frames, temporal_random_jitter=0):
    if temporal_random_jitter == 0:
        return indices

    jitter = np.random.randint(-temporal_random_jitter, temporal_random_jitter + 1, size=num_frames)
    clipped = np.clip(indices + jitter, 0, total_frames)
    return np.sort(clipped)


def load_video_av_optimized(video_path, num_frames, multi_thread_decode=False, temporal_random_jitter=0,
                            temporal_random_offset=0, use_sampling_ratio=False, **kwargs):
    """Efficiently load video frames using uniform sampling"""
    container = av.open(video_path)
    if multi_thread_decode:
        container.streams.video[0].thread_type = 'AUTO'
    video_stream = container.streams.video[0]
    total_frames = video_stream.frames
    num_frames = total_frames if num_frames == -1 else num_frames

    # Calculate timestamps for uniform sampling
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=np.int32)
    indices = temporal_random_offset_indices(indices, total_frames - 1, temporal_random_offset, use_sampling_ratio)
    indices = temporal_random_jitter_indices(indices, total_frames - 1, num_frames, temporal_random_jitter)
    logger.debug(
        f'temporal_random_jitter = {temporal_random_jitter}, temporal_random_offset = {temporal_random_offset}, use_sampling_ratio = {use_sampling_ratio}, num_frames = {num_frames}, total_frames = {total_frames}, indices = {indices}')
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
                 multi_thread_decode=False,
                 threshold=-1,
                 seed=42,
                 include_labels=None,
                 use_sampling_ratio=False):
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
        self.threshold = threshold
        self.seed = seed
        self.include_labels = include_labels
        self.use_sampling_ratio = use_sampling_ratio
        self._init_dataset()

    def _init_dataset(self):
        with open(self.annotations_path, 'rb') as f:
            self.data = json.loads(f.read())

        with open(self.vocab_path, 'rb') as f:
            self.vocab = json.loads(f.read())

        if self.threshold > 0:
            self._filter_data()

        if self.include_labels is not None:
            self._include_labels()

    def _filter_data(self):
        # convert data to { label: [ {...}, {...}.. ] }
        labels_2_vids = {label: [] for label in range(len(self.vocab))}
        for item in self.data:
            labels_2_vids[item['label']].append(item)

        # set the seed so we take the same sample everytime
        # -1 is a special case where it should be None
        seed = self.seed if self.seed != -1 else None
        rng = np.random.default_rng(seed=seed)
        data = []
        vocab = {}
        for k, v in labels_2_vids.items():
            if len(v) < self.threshold:
                continue

            # choose threshold number of videos to keep and add them to our new data
            v = rng.choice(v, self.threshold, replace=False)
            vocab[k] = self.vocab[k]
            data.extend(v)

        self.vocab = vocab
        self.data = data

    def _include_labels(self):
        self.data = list(filter(lambda x: x['label'] in self.include_labels, self.data))
        self.vocab = {k: self.vocab[k] for k in self.include_labels}

    def _read_frames(self, video_id):
        start = time.time()
        video_path = os.path.join(self.videos_path, f'{video_id}.mp4')

        frames = self.load_video(video_path,
                                 self.num_frames,
                                 self.temporal_random_jitter,
                                 self.temporal_random_offset,
                                 multithread_decode=self.multi_thread_decode,
                                 use_sampling_ratio=self.use_sampling_ratio)
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
        return frames, label, io_time, transform_time, video_id

    def __len__(self):
        return self.num_videos

    @property
    def num_videos(self):
        return len(self.data)

    @property
    def num_classes(self):
        return len(self.vocab)

    @property
    def labels(self):
        if type(self.vocab) is dict:
            return list(self.vocab.keys())
        else:
            return list(range(len(self.vocab)))

    def get_label(self, idx):
        return self.vocab[idx]

    def _transform(self, frames):
        if self.transform_fn:
            frames = self.transform_fn(frames)

        return frames

    def get_inverted_class_weights(self):
        labels = np.array(list(map(lambda x: x['label'], self.data)))
        _, counts = np.unique(labels, return_counts=True)
        # hacky way to add the class with missing weights
        counts = np.insert(counts, 30, 0)
        weights = counts / np.sum(counts)
        inverted_weights = 1 / weights
        return inverted_weights / inverted_weights.sum()
