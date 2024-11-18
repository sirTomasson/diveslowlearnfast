import time
import torch

import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from diveslowlearnfast.config import Config


def train_epoch(model: nn.Module,
                criterion: nn.Module,
                optimiser: torch.optim.Optimizer,
                loader: DataLoader,
                device,
                cfg: Config):

    loader_times = []
    batch_times = []
    loader_iter = iter(loader)
    batch_bar = tqdm(range(len(loader)), desc=f'Train batch')
    for _ in batch_bar:
        start_time = time.time()
        xb, yb, io_times, transform_times = next(loader_iter)
        loader_times.append(time.time() - start_time)

        start_time = time.time()
        yb = yb.to(device)
        xb_fast = xb[:].to(device)
        # reduce the number of frames by the alpha ratio
        # B x C x T / alpha x H x W
        xb_slow = xb[:, :, ::cfg.SLOWFAST.ALPHA].to(device)

        optimiser.zero_grad()
        o = model([xb_slow, xb_fast])
        loss = criterion(o, yb)
        loss.backward()
        batch_times.append(time.time() - start_time)

        avg_loader_time = np.mean(loader_times)
        avg_batch_time = np.mean(batch_times)
        postfix = {
            'avg_loader_time': f'{avg_loader_time:.2f}s',
            'avg_batch_time': f'{avg_batch_time:.2f}s',
            'avg_io_time': f'{io_times.numpy().mean():.2f}s',
            'avg_transform_time': f'{transform_times.numpy().mean():.2f}s',
        }
        batch_bar.set_postfix(postfix)