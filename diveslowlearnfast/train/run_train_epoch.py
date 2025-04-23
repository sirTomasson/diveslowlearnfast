import time
import torch

import numpy as np

from torch import nn, autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from diveslowlearnfast.config import Config
from diveslowlearnfast.train.multigrid import MultigridSchedule
from diveslowlearnfast.train.stats import Statistics, StatsDB

from diveslowlearnfast.train import helper as train_helper


def run_train_epoch(model: nn.Module,
                    criterion: nn.Module,
                    optimiser: torch.optim.Optimizer,
                    loader: DataLoader,
                    device,
                    cfg: Config,
                    stats_db: StatsDB,
                    epoch: int,
                    mutligrid_schedule: MultigridSchedule=None,
                    scaler: GradScaler=None):
    stats = Statistics()
    loader_iter = iter(loader)
    batch_bar = tqdm(range(len(loader)), desc='Train batch')
    n_batches_per_step = cfg.TRAIN.MACRO_BATCH_SIZE // cfg.TRAIN.BATCH_SIZE
    n_batches_per_step = n_batches_per_step if n_batches_per_step > 0 else 1
    running_loss = 0.0
    correct = 0
    count = 0
    for i in batch_bar:
        start_time = time.time()
        xb, yb, io_times, transform_times, video_ids, *_ = next(loader_iter)
        stats.update(loader_time=(time.time() - start_time))
        start_time = time.time()

        yb = yb.to(device)
        o = train_helper.forward(model, xb, device, cfg, scaler)
        loss = criterion(o, yb) / n_batches_per_step

        train_helper.backward(loss, scaler)

        running_loss += loss.item()

        with torch.no_grad():  # Add no_grad for prediction
            ypred = o.argmax(dim=-1).detach().cpu().numpy()
            ytrue = yb.detach().cpu().numpy()
            correct += (ypred == ytrue).sum()

        stats.update(
            io_time=np.mean(io_times.numpy()),
            transform_time=np.mean(transform_times.numpy()),
        )

        count += len(xb)

        # if we have completed a sufficient number of macro batches, or it is the last batch
        if (i+1) % n_batches_per_step == 0 or (i+1) == len(loader):
            if scaler:
                scaler.step(optimiser)
                scaler.update()
            else:
                optimiser.step()

            optimiser.zero_grad()

            acc = correct / count
            stats.update(accuracy=acc, loss=running_loss)
            correct = 0
            count = 0
            running_loss = 0

        stats_db.update(video_ids, ypred, ytrue, str(cfg.TRAIN.RESULT_DIR), 'train', epoch)
        stats.update(batch_time=(time.time() - start_time))

        postfix = stats.get_formatted_stats(
            'current_batch_time',
            'current_io_time',
            'current_transform_time',
            'current_loss',
            'current_accuracy',
        )
        if mutligrid_schedule:
            mutligrid_schedule.step(cfg)
            postfix['multigrid_short_cycle_crop_size'] = f'{mutligrid_schedule.get_short_cycle_crop_size(cfg)}'

        batch_bar.set_postfix(postfix)

    mean_accuracy = stats.mean_accuracy()
    mean_loss = stats.mean_loss()
    return mean_accuracy, mean_loss