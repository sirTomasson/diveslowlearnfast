import torch
import random

from torchvision.transforms.v2 import RandomCrop

from diveslowlearnfast.transforms.deterministic_random_short_side_scale import get_short_side_scale_params


def get_deterministic_transform_params(h, w, size, min_size=256, max_size=320, flip_prob=0.5):
    new_h, new_w = get_short_side_scale_params(h, w, min_size, max_size)
    return {
        'short_side_scale_params': (new_h, new_w),
        'crop_params': RandomCrop.get_params(torch.zeros((new_h, new_w)), output_size=size),
        'flip': random.random() > flip_prob,
    }
