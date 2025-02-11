from typing import Iterator

import torch

import numpy as np

from torch import nn, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from diveslowlearnfast.config import Config
from diveslowlearnfast.loss.rrr import RRRLoss
from diveslowlearnfast.train import StatsDB
from diveslowlearnfast.train import helper as train_helper
from diveslowlearnfast.train.stats import Statistics


def get_n_batches_per_step(cfg: Config):
    n_batches = cfg.TRAIN.MACRO_BATCH_SIZE // cfg.TRAIN.BATCH_SIZE
    return n_batches if n_batches > 0 else 1


def get_batch(loader: Iterator,
              device: torch.device,
              stats: Statistics,
              data_requires_grad: bool=False):

    with stats.timer('loader_time'):
        xb, yb, io_times, transform_times, video_ids, masks = next(loader)

    stats.update(
        io_time=io_times.numpy().mean(),
        transform_time=transform_times.numpy().mean()
    )
    xb = xb.to(device)
    xb.requires_grad = data_requires_grad
    return xb, yb.to(device), video_ids, masks


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


def run_train_epoch(model: nn.Module,
                    rrr_loss: RRRLoss,
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
            xb, yb, video_ids, masks = get_batch(loader_iter, device, stats, data_requires_grad=True)

            logits = train_helper.forward(model, xb, device, cfg, scaler)
            loss, _ = rrr_loss(logits, yb, model, xb, masks)
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

