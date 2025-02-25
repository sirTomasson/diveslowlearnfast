import os

import numpy as np
from tqdm import tqdm

from diveslowlearnfast.config import Config
from diveslowlearnfast.models.utils import to_slowfast_inputs


def _generate_masks(exp, percentile=50, invert=False):
    features = exp.detach().cpu().numpy()
    thresh = np.percentile(features, percentile, axis=(1, 2, 3, 4))
    thresh = thresh.reshape((-1, 1, 1, 1, 1))
    mask = features > thresh
    if invert:
        mask = ~mask
    return mask


def _save_masks(masks, video_ids, masks_cache_dir):
    for mask, video_id in zip(masks[0], video_ids):
        np.save(os.path.join(masks_cache_dir, 'slow', video_id), mask)

    for mask, video_id in zip(masks[1], video_ids):
        np.save(os.path.join(masks_cache_dir, 'fast', video_id), mask)


def _create_masks_cache_dirs_if_not_exist(cfg):
    slow_masks_dir = os.path.join(cfg.EGL.MASKS_CACHE_DIR, 'slow')
    fast_masks_dir = os.path.join(cfg.EGL.MASKS_CACHE_DIR, 'fast')
    if not os.path.exists(slow_masks_dir):
        os.makedirs(slow_masks_dir)

    if not os.path.exists(fast_masks_dir):
        os.makedirs(fast_masks_dir)


def generate_masks(loader, explainer, cfg: Config):
    _create_masks_cache_dirs_if_not_exist(cfg)

    loader = iter(loader)
    batch_bar = tqdm(range(len(loader)), desc='Generating masks')
    for _ in batch_bar:
        xb, yb, _, _, video_ids, *_ = next(loader)
        inputs = to_slowfast_inputs(
            xb,
            alpha=cfg.SLOWFAST.ALPHA,
            requires_grad=True
        )
        localisation_maps, _logits = explainer(inputs, yb)
        masks = [_generate_masks(localisation_map, invert=cfg.EGL.INVERT_MASKS, percentile=cfg.EGL.MASK_PERCENTILE) for localisation_map in localisation_maps]
        _save_masks(masks, video_ids, masks_cache_dir=cfg.EGL.MASKS_CACHE_DIR)
