import os
import torch

import numpy as np
import pandas as pd

from diveslowlearnfast.datasets.diving48 import temporal_random_offset_indices
from diveslowlearnfast.config.defaults import Config
from torch.utils.data import Dataset
from PIL import Image


class FineDivingFineDataset(Dataset):

    def __init__(self, cfg: Config, split='train', data=None, transform_fn=None):
        assert os.path.exists(cfg.DATA.DATASET_PATH)
        self.cfg = cfg
        self.dataset_root = cfg.DATA.DATASET_PATH
        self.split = split
        self.annotations = pd.read_pickle(
            os.path.join(self.dataset_root, 'Annotations', 'FineDiving_fine-grained_annotation.pkl'))
        self.actions = pd.read_pickle(os.path.join(self.dataset_root, 'Annotations', 'Sub_action_Types_Table.pkl'))
        # Add a 'None' class for zero padded frames.
        self.actions[42] = 'None'
        if data is None:
            split_filename = 'test_split.pkl' if split == 'test' else 'train_split.pkl'
            self.data = pd.read_pickle(os.path.join(self.dataset_root, 'train_test_split', split_filename))
        else:
            self.data = data

        self.videos_path = os.path.join(self.dataset_root, 'FINADiving_MTL_256s')
        self.transform_fn = transform_fn

    def _read_frames(self, video_path):
        images = np.array(sorted(os.listdir(video_path)))
        total_frames = len(images)
        indices = np.linspace(0, len(images), self.cfg.DATA.NUM_FRAMES, dtype=np.int32)
        indices = temporal_random_offset_indices(indices, total_frames, use_dynamic_temporal_stride=True,
                                                 should_wrap_around=False)
        result = []
        indices = indices[indices < total_frames]
        for image in images[indices]:
            frame = np.array(Image.open(os.path.join(video_path, image)))
            result.append(frame)

        frames = np.stack(result)
        if len(frames) < self.cfg.DATA.NUM_FRAMES:
            pad_width = [(0, self.cfg.DATA.NUM_FRAMES - len(frames)), (0, 0), (0, 0), (0, 0)]
            frames = np.pad(frames, pad_width, mode='constant', constant_values=0)

        if self.transform_fn:
            frames = self.transform_fn(frames)

        return frames, indices

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_id = self.data[idx]
        video_path = os.path.join(self.videos_path, str(video_id[0]), str(video_id[1]))
        x, indices = self._read_frames(video_path)
        y = self.annotations[video_id]['frames_labels'][indices]
        if len(y) < self.cfg.DATA.NUM_FRAMES:
            y = np.pad(y, (0, self.cfg.DATA.NUM_FRAMES - len(y)), mode='constant', constant_values=42)

        return x, torch.from_numpy(y), 0, 0, 0, False

    def get_action(self, idx):
        return self.actions[idx]

    @property
    def num_classes(self):
        return len(self.actions)
