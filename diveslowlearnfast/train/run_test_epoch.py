import time
import torch

import numpy as np

from torch import nn, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from diveslowlearnfast.config import Config
from diveslowlearnfast.egl import GradCamExplainer
from diveslowlearnfast.train.stats import StatsDB
from diveslowlearnfast.eval import metrics
from diveslowlearnfast.train import helper as train_helper

def calculate_dice_factors(cfg: Config, attn_maps, masks):
    masks = [masks[:, :, ::cfg.SLOWFAST.ALPHA], masks[:]]
    dice_factors = []
    for attn_map, mask in zip(attn_maps, masks):
        attn_map = attn_map.detach().cpu()
        mask = mask.detach().cpu()
        dice_factor = metrics.dice_factor(mask, attn_map)
        dice_factors.append(dice_factor.mean().item())

    return dice_factors


def calculate_iou(cfg, attn_maps, masks):
    masks = [masks[:, :, ::cfg.SLOWFAST.ALPHA], masks[:]]
    ious = []
    for attn_map, mask in zip(attn_maps, masks):
        attn_map = attn_map.detach().cpu()
        mask = mask.detach().cpu()
        iou = metrics.iou(mask, attn_map)
        ious.append(iou.mean().item())
    return ious


def get_default_metrics_dict():
    return {
        'accuracies': [],
        'iou_slow': [],
        'iou_fast': [],
        'dice_slow': [],
        'dice_fast': [],
    }

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
    metrics_dict = get_default_metrics_dict()
    for _ in batch_bar:
        start_time = time.time()
        xb, yb, io_times, transform_times, video_ids, m = next(loader_iter)
        xb.requires_grad = True
        loader_times.append(time.time() - start_time)

        start_time = time.time()
        o = train_helper.forward(model, xb, device, cfg, scaler)
        if cfg.EGL.ENABLED:
            logits = o[1]
            if cfg.IOU_METRICS.ENABLED:
                dice_factors = calculate_dice_factors(cfg, o[0], m)
                ious = calculate_iou(cfg, o[0], m)
                metrics_dict['dice_slow'].append(dice_factors[0])
                metrics_dict['dice_fast'].append(dice_factors[1])
                metrics_dict['iou_slow'].append(ious[0])
                metrics_dict['iou_fast'].append(ious[1])
        else:
            logits = o

        ypred = logits.argmax(dim=-1).detach().cpu().numpy()
        ytrue = yb.numpy()
        acc = (ytrue == ypred).sum() / (len(yb))
        metrics_dict['accuracies'].append(acc)
        batch_times.append(time.time() - start_time)

        stats_db.update(video_ids, ypred, ytrue, str(cfg.TRAIN.RESULT_DIR), 'test', epoch)

        avg_loader_time = np.mean(loader_times)
        avg_batch_time = np.mean(batch_times)
        postfix = {
            'loader_time': f'{avg_loader_time:.2f}s',
            'io_time': f'{io_times.numpy().mean():.2f}s',
            'transform_time': f'{transform_times.numpy().mean():.2f}s',
            'batch_time': f'{avg_batch_time:.2f}s',
            'acc': f'{np.mean(metrics_dict["accuracies"]):.3f}',
        }
        batch_bar.set_postfix(postfix)

    return metrics_dict