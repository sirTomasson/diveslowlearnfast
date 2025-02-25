import os
import shutil

import torch
import logging

from torch.utils.data import DataLoader

from diveslowlearnfast.config import Config
from diveslowlearnfast.datasets import Diving48Dataset
from diveslowlearnfast.egl.explainer import ExplainerStrategy
from diveslowlearnfast.egl.generate_masks import generate_masks
from diveslowlearnfast.train import StatsDB
from diveslowlearnfast.train.helper import get_train_transform
from diveslowlearnfast.train.stats import wps_strategy

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'ERROR'))

logger = logging.getLogger(__name__)

def get_difficult_video_ids(stats_db: StatsDB, epoch, cfg: Config):
    if len(cfg.EGL.RUN_ID) > 0:
        run_id = cfg.EGL.RUN_ID
    else:
        run_id = str(cfg.TRAIN.RESULT_DIR)

    strategy = wps_strategy(stats_db, cfg)
    logger.info(f'strategy = {cfg.EGL.WORST_PERFORMER_STRATEGY}')
    difficult_samples = strategy(
        epoch_start=(epoch - 10),
        run_id=run_id,
        split='train'
    )

    return list(map(lambda x: x[0], difficult_samples))


def purge_masks_cache(masks_cache_dir):
    if os.path.exists(masks_cache_dir):
        shutil.rmtree(masks_cache_dir)


def augment_samples(model, videos_ids, cfg: Config, device: torch.device):
    train_loader, _ = get_difficult_samples_loader_and_dataset(cfg, videos_ids)
    explainer = ExplainerStrategy.get_explainer(model, cfg, device)
    generate_masks(train_loader, explainer, cfg)


def get_difficult_samples_loader_and_dataset(cfg, video_ids):
    train_transform = get_train_transform(cfg)

    train_dataset = Diving48Dataset(
        cfg.DATA.DATASET_PATH,
        cfg.DATA.NUM_FRAMES,
        alpha=cfg.SLOWFAST.ALPHA,
        dataset_type='train',
        transform_fn=train_transform,
        use_decord=cfg.DATA_LOADER.USE_DECORD,
        temporal_random_jitter=cfg.DATA.TEMPORAL_RANDOM_JITTER,
        temporal_random_offset=cfg.DATA.TEMPORAL_RANDOM_OFFSET,
        multi_thread_decode=cfg.DATA.MULTI_THREAD_DECODE,
        threshold=cfg.DATA.THRESHOLD,
        use_sampling_ratio=cfg.DATA.USE_SAMPLING_RATIO,
        video_ids=video_ids,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        shuffle=True,
    )

    return train_loader, train_dataset
