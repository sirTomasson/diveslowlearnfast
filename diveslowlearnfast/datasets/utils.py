import os
import av

import numpy as np

from PIL import Image

def get_sample(dataset):
    result = next(iter(dataset))
    return result[0], result[1]


def read_video_jpeg(path):
    video = []
    filenames = sorted(os.listdir(path))
    for filename in filenames:
        image_path = f'{path}/{filename}'
        if not os.path.exists(image_path):
            break
        img = Image.open(image_path)
        video.append(np.array(img))
    return np.stack(video)

def read_video_mp4(path, multithread_decode=True):
    frames = []
    container = av.open(path)
    if multithread_decode:
        container.streams.video[0].thread_type = 'AUTO'

    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format='rgb24'))

    return np.stack(frames)

def read_segmentation_mask(path):
    return read_video_jpeg(path)


def read_diver_segmentation_mask(path):
    masks = read_segmentation_mask(path)
    return np.stack([find_diver_mask(mask) for mask in masks])


def find_diver_mask(mask):
    w = mask.shape[1]
    mw = w//6
    mask_center = mask[:, w//2-mw:w//2+mw]
    masses = np.bincount(mask_center.flatten())
    masses = masses[1:min(4, len(masses) - 1)]
    if len(masses) == 0:
        return np.zeros_like(mask)
    return mask == np.argmax(masses) + 1

