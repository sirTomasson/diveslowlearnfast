import os
import av

import numpy as np

from PIL import Image

from diveslowlearnfast.memfs_cache import get_cache_instance


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


def read_video_from_image_indices(path, indices, format='jpg', dtype=np.float32):
    assert format in ['jpg', 'png']
    video = []
    for idx in indices:
        image_path = f'{path}/{idx:04d}.{format}'
        if not os.path.exists(image_path):
            if len(video) == 0:
                raise Exception(f'Empty sequence for {image_path}.')
            # If the image does not exist we got an indice beyond the last image in the sequence, so we should insert
            # a black frame
            video.append(np.zeros(video[-1].shape, dtype=dtype))
            continue

        with get_cache_instance().open(image_path) as f:
            img = Image.open(f)
            video.append(np.array(img, dtype=dtype))

    return np.stack(video, dtype=dtype)

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


def read_diver_segmentation_mask(path, indices=None):
    if indices is None:
        masks = read_segmentation_mask(path)
    else:
        masks = read_video_from_image_indices(path, indices, 'png', dtype=np.uint8)

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

