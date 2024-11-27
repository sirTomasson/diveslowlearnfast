import time
import torch

import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from diveslowlearnfast.config import Config


def run_train_epoch(model: nn.Module,
                    criterion: nn.Module,
                    optimiser: torch.optim.Optimizer,
                    loader: DataLoader,
                    device,
                    cfg: Config):

    loader_times = []
    batch_times = []
    loader_iter = iter(loader)
    batch_bar = tqdm(range(len(loader)), desc='Train batch')
    accuracies = []
    losses = []
    n_macro_batches = cfg.TRAIN.MACRO_BATCH_SIZE // cfg.TRAIN.BATCH_SIZE
    loss = 0
    correct = 0
    count = 0
    for i in batch_bar:
        start_time = time.time()
        xb, yb, io_times, transform_times = next(loader_iter)
        loader_times.append(time.time() - start_time)
        start_time = time.time()

        yb = yb.to(device)
        xb_fast = xb[:].to(device)
        # reduce the number of frames by the alpha ratio
        # B x C x T / alpha x H x W
        xb_slow = xb[:, :, ::cfg.SLOWFAST.ALPHA].to(device)

        o = model([xb_slow, xb_fast])
        current_loss = criterion(o, yb)
        loss += current_loss.item()
        current_loss = current_loss / n_macro_batches
        current_loss.backward()  # Backward pass without clearing gradients

        with torch.no_grad():  # Add no_grad for prediction
            ypred = o.argmax(dim=-1)
            correct += (yb == ypred).cpu().numpy().sum()

        avg_loader_time = np.mean(loader_times)
        postfix = {
            'loader_time': f'{avg_loader_time:.3f}s',
            'io_time': f'{io_times.numpy().mean():.3f}s',
            'transform_time': f'{transform_times.numpy().mean():.3f}s',
        }

        count += len(xb)

        # if we have completed a sufficient number of macro batches, or it is the last batch
        if (i+1) % n_macro_batches == 0 or (i+1) == len(loader):
            optimiser.step()
            optimiser.zero_grad()

            acc = correct / count
            accuracies.append(acc)
            loss /= n_macro_batches
            losses.append(loss)
            postfix['loss'] = f'{loss:.3f}'
            postfix['acc'] = f'{acc:.3f}'
            correct = 0
            count = 0
            loss = 0


        batch_times.append(time.time() - start_time)
        avg_batch_time = np.mean(batch_times)
        postfix['batch_time'] = f'{avg_batch_time:.3f}s'

        batch_bar.set_postfix(postfix)

    return np.mean(accuracies), np.mean(losses)