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
    running_acc = None
    running_loss = None
    for _ in batch_bar:
        start_time = time.time()
        xb, yb, io_times, transform_times = next(loader_iter)
        # loader_times.append(time.time() - start_time)
        #
        # start_time = time.time()
        # yb = yb.to(device)
        # xb_fast = xb[:].to(device)
        # # reduce the number of frames by the alpha ratio
        # # B x C x T / alpha x H x W
        # xb_slow = xb[:, :, ::cfg.SLOWFAST.ALPHA].to(device)
        #
        # optimiser.zero_grad()
        # o = model([xb_slow, xb_fast])
        # loss = criterion(o, yb)
        # loss.backward()
        # ypred = o.argmax(dim=-1)
        # correct = (yb == ypred).cpu().detach().numpy().sum()
        # acc = correct / len(yb)
        # running_acc = acc if running_acc is None else (running_acc + acc) / 2
        # running_loss = acc if running_loss is None else (running_loss + loss.item()) / 2
        batch_times.append(time.time() - start_time)

        avg_loader_time = np.mean(loader_times)
        avg_batch_time = np.mean(batch_times)
        postfix = {
            'loader_time': f'{avg_loader_time:.2f}s',
            # 'io_time': f'{io_times.numpy().mean():.2f}s',
            # 'transform_time': f'{transform_times.numpy().mean():.2f}s',
            'batch_time': f'{avg_batch_time:.2f}s',
            # 'acc': f'{acc:.3f}',
            # 'loss': f'{loss.item():.3f}',
        }
        batch_bar.set_postfix(postfix)

    return running_acc, running_loss