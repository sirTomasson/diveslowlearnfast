import time
import torch

import numpy as np

from torch import nn, autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from diveslowlearnfast.config import Config
from diveslowlearnfast.train.multigrid import MultigridSchedule
from diveslowlearnfast.train.stats import Statistics


def run_train_epoch(model: nn.Module,
                    criterion: nn.Module,
                    optimiser: torch.optim.Optimizer,
                    loader: DataLoader,
                    device,
                    cfg: Config,
                    mutligrid_schedule: MultigridSchedule=None,
                    scaler: GradScaler=None):
    stats = Statistics()
    loader_iter = iter(loader)
    batch_bar = tqdm(range(len(loader)), desc='Train batch')
    n_macro_batches = cfg.TRAIN.MACRO_BATCH_SIZE // cfg.TRAIN.BATCH_SIZE
    n_macro_batches = n_macro_batches if n_macro_batches > 0 else 1
    loss = 0
    correct = 0
    count = 0
    for i in batch_bar:
        start_time = time.time()
        xb, yb, io_times, transform_times = next(loader_iter)
        stats.update(loader_time=(time.time() - start_time))
        start_time = time.time()

        yb = yb.to(device)
        xb_fast = xb[:].to(device)
        # reduce the number of frames by the alpha ratio
        # B x C x T / alpha x H x W
        xb_slow = xb[:, :, ::cfg.SLOWFAST.ALPHA].to(device)

        if scaler:
            with autocast(device_type='cuda', dtype=torch.float16):
                o = model([xb_slow, xb_fast])
                current_loss = criterion(o, yb)
        else:
            o = model([xb_slow, xb_fast])
            current_loss = criterion(o, yb)


        loss += current_loss.item()
        current_loss = current_loss / n_macro_batches

        # Backward pass without clearing gradients
        if scaler:
            scaler.scale(current_loss).backward()
        else:
            current_loss.backward()

        with torch.no_grad():  # Add no_grad for prediction
            ypred = o.argmax(dim=-1)
            correct += (yb == ypred).cpu().numpy().sum()

        stats.update(
            io_time=np.mean(io_times.numpy()),
            transform_time=np.mean(transform_times.numpy()),
        )

        count += len(xb)

        # if we have completed a sufficient number of macro batches, or it is the last batch
        if (i+1) % n_macro_batches == 0 or (i+1) == len(loader):
            if scaler:
                scaler.step(optimiser)
                scaler.update()

            optimiser.step()
            optimiser.zero_grad()

            acc = correct / count
            loss /= n_macro_batches
            stats.update(accuracy=acc, loss=loss)
            correct = 0
            count = 0
            loss = 0

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