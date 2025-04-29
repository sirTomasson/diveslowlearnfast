import json
import math
import random
import av
import os
import time
import logging
import torch

import numpy as np

from torch.utils.data import Dataset

from diveslowlearnfast.datasets.superimpose_confounder import superimpose_confounder
from PIL import Image


logger = logging.getLogger(__name__)

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'ERROR'))


def extend_under_represented_classes(data):
    labels = list(map(lambda x: x['label'], data))

    unique, counts = np.unique(labels, return_counts=True)
    max_count = np.max(counts)

    new_data = []
    start = 0
    for count in counts:
        if count >= max_count:
            new_data.extend(data[start:start + count])
            start += count
            continue

        new_data.extend(data[start:start + count] * (max_count // count))
        start += count

    return new_data


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


def wrap_around(offset_indices, total_frames, min=0):
    result = []
    for idx in offset_indices:
        if idx > total_frames:
            diff = abs(offset_indices[-1] - offset_indices[-2])
            result.insert(min, result[0] - diff)
        else:
            result.append(idx)

    return np.clip(result, min, total_frames)


def temporal_random_offset_indices(indices, total_frames, temporal_random_offset=0, use_sampling_ratio=False,
                                   should_wrap_around=True):
    if temporal_random_offset == 0 and not use_sampling_ratio:
        return indices

    if use_sampling_ratio:
        # ratio between total number frames and number of frames being sampled
        # total_frames/num_frames is the minimum by which we need to shift the indices in order to cover all frames
        # across different epochs. Calculating this dynamically may result in more stable behaviour.
        temporal_random_offset = math.floor(total_frames / len(indices))
        logger.debug(
            f"use_sampling_ratio = True, calculating temporal_random_offset: {total_frames}/{len(indices)}={temporal_random_offset}")

    offset_indices = math.floor(random.uniform(min(indices), temporal_random_offset)) + indices
    if should_wrap_around:
        offset_indices = wrap_around(offset_indices, total_frames, min(indices))

    return offset_indices


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

def read_video_from_image_indices(path, indices, format='jpg'):
    assert format in ['jpg', 'png']
    video = []
    for idx in indices:
        image_path = f'{path}/{idx:04d}.{format}'
        if not os.path.exists(image_path):
            if len(video) == 0:
                raise Exception(f'Empty sequence for {image_path}.')
            # If the image does not exist we got an indice beyond the last image in the sequence, so we should insert
            # a black frame
            video.append(np.zeros(video[-1].shape))
            continue

        img = Image.open(image_path)
        video.append(np.array(img))

    return np.stack(video)


def load_video_from_images(video_path, num_frames, temporal_random_jitter=0, temporal_random_offset=0,
                           use_sampling_ratio=False, **kwargs):
    """Efficiently load video frames using uniform sampling"""
    total_frames = len(os.listdir(video_path))

    indices = np.linspace(1, total_frames, num_frames, dtype=np.int32)
    indices = temporal_random_offset_indices(indices, total_frames, temporal_random_offset, use_sampling_ratio,
                                             should_wrap_around=False)
    indices = temporal_random_jitter_indices(indices, total_frames, num_frames, temporal_random_jitter)
    return read_video_from_image_indices(video_path, indices, 'jpg')


def collate_fn(batch):
    max_frames = max(video.shape[0] for video in batch)
    return [pad_video(video, max_frames) for video in batch]


def get_video_loader(use_decord, loader_mode):
    if loader_mode == 'mp4':
        if use_decord:
            return decord_load_video
        else:
            return load_video_av_optimized
    elif loader_mode == 'jpg':
        return load_video_from_images
    else:
        raise ValueError(f'Unknown loader mode: {loader_mode}')


def get_videos_dir(loader_mode):
    if loader_mode == 'mp4':
        return 'rgb'
    elif loader_mode == 'jpg':
        return 'JPEGImages'
    raise ValueError(f'Unknown loader mode: {loader_mode}')


class Diving48Dataset(Dataset):

    def __init__(self,
                 dataset_path,
                 num_frames,
                 alpha=1,
                 dataset_type='train',
                 dataset_version='V2',
                 temporal_random_jitter=0,
                 temporal_random_offset=0,
                 transform_fn=None,
                 target_fps=None,
                 use_decord=False,
                 multi_thread_decode=False,
                 loader_mode='mp4',
                 threshold=-1,
                 seed=42,
                 include_labels=None,
                 use_sampling_ratio=False,
                 video_ids=None,
                 masks_cache_dir=None,
                 extend_classes=False,
                 mask_type=None,
                 should_wrap_around=True):
        super().__init__()
        assert loader_mode in ['mp4', 'jpg']
        assert dataset_type in ['train', 'test']
        self.videos_path = os.path.join(dataset_path, get_videos_dir(loader_mode))
        self.annotations_path = os.path.join(dataset_path, f'Diving48_{dataset_version}_{dataset_type}.json')
        self.vocab_path = os.path.join(dataset_path, f'Diving48_vocab.json')
        self.num_frames = num_frames
        self.alpha = alpha
        self.target_fps = target_fps
        self.transform_fn = transform_fn
        self.temporal_random_jitter = temporal_random_jitter
        self.temporal_random_offset = temporal_random_offset
        self.load_video = get_video_loader(use_decord, loader_mode)
        self.loader_mode = loader_mode
        self.multi_thread_decode = multi_thread_decode
        self.threshold = threshold
        self.seed = seed
        self.include_labels = include_labels
        self.use_sampling_ratio = use_sampling_ratio
        self.include_video_ids = video_ids
        self.masks_cache_dir = masks_cache_dir
        self.extend_classes = extend_classes
        self.mask_type = mask_type
        self.should_wrap_around = should_wrap_around
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

        if self.include_video_ids is not None:
            self.data = list(filter(lambda video: video['vid_name'] in self.include_video_ids, self.data))

        if self.extend_classes:
            self.data = extend_under_represented_classes(self.data)

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
        filename = f'{video_id}.mp4' if self.loader_mode == 'mp4' else video_id
        video_path = os.path.join(self.videos_path, filename)

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

    def _create_confounder_masks(self, label, size):
        _, t, h, w = size
        mask = torch.zeros((1, t, h, w), dtype=torch.bool)
        mask = superimpose_confounder(mask, label, binary=True)
        return mask[:, ::self.alpha], mask[:]

    def _read_mask(self, video_id, h, w):
        mask_slow_filename = os.path.join(self.masks_cache_dir, 'slow', f'{video_id}.npy')
        mask_fast_filename = os.path.join(self.masks_cache_dir, 'fast', f'{video_id}.npy')
        if os.path.exists(mask_slow_filename) and os.path.exists(mask_fast_filename):
            return np.load(mask_slow_filename), np.load(mask_fast_filename)

        return np.zeros((1, self.num_frames // self.alpha, h, w), dtype=np.bool_), np.zeros((1, self.num_frames, h, w),
                                                                                            dtype=np.bool_)

    def _get_mask(self, label, video_id, size):
        if self.mask_type == 'confounder':
            return self._create_confounder_masks(label, size)
        else:
            _, _, h, w = size
            return self._read_mask(video_id, h, w)

    def __getitem__(self, index):
        video_id = self.data[index]['vid_name']
        label = self.data[index]['label']
        frames, io_time, transform_time = self._read_frames(video_id)
        if self.mask_type is not None:
            mask_slow, mask_fast = self._get_mask(label, video_id, frames.shape)
            return frames, label, io_time, transform_time, video_id, mask_slow, mask_fast

        return frames, label, io_time, transform_time, video_id, False, False

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
        inverted_weights = 1 - weights
        return inverted_weights / inverted_weights.sum()
