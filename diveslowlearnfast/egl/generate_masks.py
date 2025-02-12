import os

import numpy as np
from tqdm import tqdm

from diveslowlearnfast.config import Config


def _generate_masks(exp, percentile=50):
    features = exp.numpy()
    thresh = np.percentile(features, percentile, axis=(1, 2, 3, 4))
    thresh = thresh.reshape((-1, 1, 1, 1, 1))
    return ~(features > thresh)


def _save_masks(masks, video_ids, masks_cache_dir):
    for mask, video_id in zip(masks, video_ids):
        np.save(os.path.join(masks_cache_dir, video_id), mask)


def generate_masks(loader, explainer, cfg: Config):
    if not os.path.exists(cfg.EGL.MASKS_CACHE_DIR):
        os.makedirs(cfg.EGL.MASKS_CACHE_DIR)

    loader = iter(loader)
    batch_bar = tqdm(range(len(loader)), desc='Generating masks')
    for _ in batch_bar:
        xb, yb, _, _, video_ids, _ = next(loader)
        exp = explainer(xb, yb)
        masks = _generate_masks(exp)
        _save_masks(masks, video_ids, masks_cache_dir=cfg.EGL.MASKS_CACHE_DIR)
