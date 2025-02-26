import logging
import os

import torch

import numpy as np

from torch import nn, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from diveslowlearnfast.config import Config
from diveslowlearnfast.loss.rrr import RRRLoss, DualPathRRRLoss
from diveslowlearnfast.models.utils import to_slowfast_inputs
from diveslowlearnfast.train import StatsDB
from diveslowlearnfast.train import helper as train_helper
from diveslowlearnfast.train.stats import Statistics

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'ERROR'))

logger = logging.getLogger(__name__)

def get_n_batches_per_step(cfg: Config):
    n_batches = cfg.TRAIN.MACRO_BATCH_SIZE // cfg.TRAIN.BATCH_SIZE
    return n_batches if n_batches > 0 else 1


def should_step(batch_idx: int, n_batches: int, loader: DataLoader):
    return (batch_idx+1) % n_batches == 0 or (batch_idx+1) == len(loader)

def step(optimiser: torch.optim.Optimizer, scaler: GradScaler=None):
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
    stats_db.add_loss(losses['total_loss'], epoch, 'loss', str(cfg.TRAIN.RESULT_DIR), 'train')
    stats_db.add_loss(losses['gradient_loss_path_0'], epoch, 'slow', str(cfg.TRAIN.RESULT_DIR), 'train')
    stats_db.add_loss(losses['gradient_loss_path_1'], epoch, 'fast', str(cfg.TRAIN.RESULT_DIR), 'train')
    stats_db.add_loss(losses['ce_loss'], epoch, 'ce', str(cfg.TRAIN.RESULT_DIR), 'train')


def run_train_epoch(model: nn.Module,
                    rrr_loss: RRRLoss | DualPathRRRLoss,
                    optimiser: torch.optim.Optimizer,
                    loader: DataLoader,
                    device,
                    cfg: Config,
                    stats_db: StatsDB,
                    epoch: int,
                    scaler: GradScaler = None):
    batch_bar = tqdm(range(len(loader)), desc='Train EGL batch')
    n_batches_per_step = get_n_batches_per_step(cfg)
    loader_iter = iter(loader)
    stats = Statistics()
    Y_true = []
    Y_pred = []
    running_loss = 0.0
    for i in batch_bar:
        with stats.timer('batch_time'):
            xb, yb, video_ids, masks_slow, masks_fast = train_helper.get_batch(
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

            logits = model(inputs)
            loss, losses = rrr_loss(logits, yb, inputs, [masks_slow, masks_fast])
            add_losses_entry(stats_db, losses, epoch, cfg)
            loss /= n_batches_per_step # scale loss by the number of batches in a step
            train_helper.backward(loss)

            running_loss += loss.item()

            Y_pred.extend(logits.argmax(dim=-1).detach().cpu().tolist())
            Y_true.extend(yb.detach().cpu().tolist())

            if should_step(i, n_batches_per_step, loader):
                step(optimiser, scaler)

                stats_db.update(video_ids, Y_pred, Y_true, str(cfg.TRAIN.RESULT_DIR), 'train', epoch)
                stats.update(
                    accuracy=calc_accuracy(Y_true, Y_pred),
                    loss=running_loss
                )

                running_loss = 0.0; Y_true = []; Y_pred = []

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

