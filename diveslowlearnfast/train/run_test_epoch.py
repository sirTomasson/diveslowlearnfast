import time
import torch

import numpy as np

from torch import nn, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from diveslowlearnfast.config import Config
from diveslowlearnfast.train import helper as train_helper
from diveslowlearnfast.train.stats import StatsDB


@torch.no_grad()
def run_test_epoch(model: nn.Module,
              loader: DataLoader,
              device,
              cfg: Config,
              stats_db: StatsDB,
              epoch: int,
              scaler: GradScaler=None):

    loader_times = []
    batch_times = []
    loader_iter = iter(loader)
    batch_bar = tqdm(range(len(loader)), desc='Test batch')
    accuracies = []
    for _ in batch_bar:
        start_time = time.time()
        xb, yb, io_times, transform_times, video_ids, *_ = next(loader_iter)
        loader_times.append(time.time() - start_time)

        start_time = time.time()
        o = train_helper.forward(model, xb, device, cfg, scaler)
        ypred = o.argmax(dim=-1).detach().cpu().numpy()
        ytrue = yb.numpy()
        acc = (ytrue == ypred).sum() / (len(yb))
        accuracies.append(acc)
        batch_times.append(time.time() - start_time)

        stats_db.update(video_ids, ypred, ytrue, str(cfg.TRAIN.RESULT_DIR), 'test', epoch)

        avg_loader_time = np.mean(loader_times)
        avg_batch_time = np.mean(batch_times)
        postfix = {
            'loader_time': f'{avg_loader_time:.2f}s',
            'io_time': f'{io_times.numpy().mean():.2f}s',
            'transform_time': f'{transform_times.numpy().mean():.2f}s',
            'batch_time': f'{avg_batch_time:.2f}s',
            'acc': f'{acc:.3f}',
        }
        batch_bar.set_postfix(postfix)

    return np.mean(accuracies)