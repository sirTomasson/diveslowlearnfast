import logging
import os
import random
import torch

import numpy as np

from torch import nn, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from diveslowlearnfast.config import Config
from diveslowlearnfast.egl.generate_masks import generate_masks_from_localisation_maps
from diveslowlearnfast.models.utils import to_slowfast_inputs
from diveslowlearnfast.train import helper as train_helper
from diveslowlearnfast.train.stats import Statistics, StatsDB

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'ERROR'))

logger = logging.getLogger(__name__)


def get_n_batches_per_step(cfg: Config):
    n_batches = cfg.TRAIN.MACRO_BATCH_SIZE // cfg.TRAIN.BATCH_SIZE
    return n_batches if n_batches > 0 else 1


def should_step(batch_idx: int, n_batches: int, loader: DataLoader):
    return (batch_idx + 1) % n_batches == 0 or (batch_idx + 1) == len(loader)


def step(optimiser: torch.optim.Optimizer, scaler: GradScaler = None):
    if scaler:
        scaler.step(optimiser)
        scaler.update()

    optimiser.step()
    optimiser.zero_grad()


def calc_accuracy(Y_true, Y_pred):
    Y_true = np.array(Y_true)
    Y_pred = np.array(Y_pred)
    return (Y_true == Y_pred).sum() / len(Y_true)


def add_losses_entry(stats_db: StatsDB, losses, epoch, cfg: Config):
    if cfg.EGL.LOSS_FUNC in ['rrr', 'rrr_v2']:
        stats_db.add_loss(losses['total_loss'], epoch, 'loss', str(cfg.TRAIN.RESULT_DIR), 'train')
        stats_db.add_loss(losses['gradient_loss_path_0'], epoch, 'slow', str(cfg.TRAIN.RESULT_DIR), 'train')
        stats_db.add_loss(losses['gradient_loss_path_1'], epoch, 'fast', str(cfg.TRAIN.RESULT_DIR), 'train')
        stats_db.add_loss(losses['ce_loss'], epoch, 'ce', str(cfg.TRAIN.RESULT_DIR), 'train')
    elif cfg.EGL.LOSS_FUNC == 'dice':
        stats_db.add_loss(losses['total_loss'], epoch, 'loss', str(cfg.TRAIN.RESULT_DIR), 'train')
        stats_db.add_loss(losses['dice_loss'], epoch, 'dice', str(cfg.TRAIN.RESULT_DIR), 'train')
        stats_db.add_loss(losses['ce_loss'], epoch, 'ce', str(cfg.TRAIN.RESULT_DIR), 'train')


def get_masks(localisation_maps, cfg, device, indices=None):
    if cfg.EGL.METHOD in ['confounder', 'ogl', 'cache']:
        return [map.to(device) for map in localisation_maps]


    masks = generate_masks_from_localisation_maps(localisation_maps, cfg, indices)
    return [torch.from_numpy(masks_slow_and_fast).to(device) for masks_slow_and_fast in masks]


def get_mask_indices(cfg, targets, logits, video_ids, hard_video_ids):
    if cfg.EGL.WORST_PERFORMER_STRATEGY in ['percentile', 'median']:
        assert hard_video_ids is not None
        assert video_ids is not None
        indices = np.array([video_id in hard_video_ids for video_id in video_ids])
    elif cfg.EGL.WORST_PERFORMER_STRATEGY == 'classify':
        y_pred = torch.softmax(logits, dim=-1).argmax(dim=-1)
        indices = (targets != y_pred).detach().cpu().numpy()
    else:
        indices = np.ones(targets.shape, dtype=np.bool_)

    return indices

def get_loss_params(cfg: Config,
                    localisation_maps,
                    inputs,
                    targets,
                    logits,
                    masks=None,
                    hard_video_ids=None,
                    video_ids=None):

    if cfg.EGL.METHOD == 'cache':
        assert masks is not None
        masks_from_localisation_maps = get_masks(masks, cfg, logits.device)
    else:
        indices = get_mask_indices(cfg, targets, logits, video_ids, hard_video_ids)
        masks_from_localisation_maps = get_masks(localisation_maps, cfg, logits.device, indices)

    if cfg.EGL.METHOD == 'ogl' and cfg.EGL.LOSS_FUNC in ['rrr', 'rrr_v2']:
        # In this mode we'll invert the diver masks so the model will be penalised for high gradients were there is no
        # diver. In a subtle way kinda of similar to the CUTOUT_TRANSFORM method.
        if random.random() <= cfg.CUTOUT_SEGMENT.PROB:
            masks = ~masks
        else:
            masks = torch.zeros_like(masks)

        return {
            'logits': logits,
            'targets': targets,
            'masks': [masks[:, :, ::cfg.SLOWFAST.ALPHA].to(logits.device), masks[:].to(logits.device)],
            'inputs': inputs,
        }
    elif cfg.EGL.LOSS_FUNC in ['rrr', 'rrr_v2']:
        return {
            'logits': logits,
            'targets': targets,
            'masks': masks_from_localisation_maps,
            'inputs': inputs,
        }
    elif cfg.EGL.LOSS_FUNC == 'dice':
        assert masks is not None, 'Masks should be provided for Dice loss'
        return {
            'logits': logits,
            'targets': targets,
            'gradcams': masks_from_localisation_maps,
            'masks': [masks[:, :, ::cfg.SLOWFAST.ALPHA].to(logits.device), masks[:].to(logits.device)],
        }
    else:
        raise ValueError(f'Unknown loss function: {cfg.EGL.LOSS_FUNC}')



def run_train_epoch(model: nn.Module,
                    loss_fn: nn.Module,
                    optimiser: torch.optim.Optimizer,
                    loader: DataLoader,
                    device,
                    cfg: Config,
                    stats_db: StatsDB,
                    epoch: int,
                    scaler: GradScaler = None,
                    hard_video_ids=None):
    if hard_video_ids is None:
        hard_video_ids = []
    hard_video_ids = set(hard_video_ids)

    batch_bar = tqdm(range(len(loader)), desc='Train EGL batch')
    n_batches_per_step = get_n_batches_per_step(cfg)
    loader_iter = iter(loader)
    stats = Statistics()
    Y_true = []
    Y_pred = []
    V_ids = []
    running_loss = 0.0
    for i in batch_bar:
        with stats.timer('batch_time'):
            xb, yb, video_ids, masks = train_helper.get_batch(
                loader_iter,
                device,
                stats
            )

            inputs = to_slowfast_inputs(
                xb,
                alpha=cfg.SLOWFAST.ALPHA,
                requires_grad=True,
                device=device
            )

            # 'model' is actually an explainer
            localisation_maps, logits = model(inputs)

            params = get_loss_params(cfg,
                                     localisation_maps,
                                     inputs,
                                     yb,
                                     logits,
                                     masks=masks,
                                     hard_video_ids=hard_video_ids,
                                     video_ids=video_ids)
            loss, losses = loss_fn(**params)
            add_losses_entry(stats_db, losses, epoch, cfg)
            loss /= n_batches_per_step  # scale loss by the number of batches in a step
            train_helper.backward(loss)

            running_loss += loss.item()

            y_pred = torch.softmax(logits, dim=-1).argmax(dim=-1)
            Y_pred.extend(y_pred.detach().cpu().tolist())
            Y_true.extend(yb.detach().cpu().tolist())
            V_ids.extend(video_ids)

            if should_step(i, n_batches_per_step, loader):
                step(optimiser, scaler)
                stats.update(
                    accuracy=calc_accuracy(Y_true, Y_pred),
                    loss=running_loss
                )
                stats_db.update(V_ids, Y_pred, Y_true, str(cfg.TRAIN.RESULT_DIR), 'train', epoch)

                running_loss = 0.0
                Y_true = []
                Y_pred = []
                V_ids = []

        batch_bar.set_postfix(stats.get_formatted_stats(
            'current_batch_time',
            'current_io_time',
            'current_transform_time',
            'current_loss',
            'current_accuracy',
        ))

    mean_accuracy = stats.mean_accuracy()
    mean_loss = stats.mean_loss()
    return mean_accuracy, mean_loss
