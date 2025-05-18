import os
import torch

import numpy as np

from torch import nn, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from diveslowlearnfast.config import Config
from diveslowlearnfast.train import helper as train_helper

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from diveslowlearnfast.train.run_test_epoch import calculate_dice_factors, calculate_iou
from diveslowlearnfast.train.stats import StatsDB



def run_eval_epoch(model: nn.Module,
                   loader: DataLoader,
                   device,
                   cfg: Config,
                   labels,
                   stats,
                   stats_db: StatsDB,
                   scaler: GradScaler = None):
    loader_iter = iter(loader)
    eval_bar = tqdm(range(len(loader)), desc='Eval batch')
    Y_true = []
    Y_pred = []
    V_ids = []
    iou_metrics = {
        'dice_slow': [],
        'dice_fast': [],
        'iou_slow': [],
        'iou_fast': [],
    }
    for _ in eval_bar:
        xb, yb, io_times, transform_times, video_ids, m = next(loader_iter)
        o = train_helper.forward(model, xb, device, cfg, scaler)
        if cfg.EGL.ENABLED:
            logits = o[1]
            if cfg.IOU_METRICS.ENABLED:
                dice_factors = calculate_dice_factors(cfg, o[0], m)
                ious = calculate_iou(cfg, o[0], m)
                iou_metrics['dice_slow'].append(dice_factors[0])
                iou_metrics['dice_fast'].append(dice_factors[1])
                iou_metrics['iou_slow'].append(ious[0])
                iou_metrics['iou_fast'].append(ious[1])
        else:
            logits = o

        ypred = logits.argmax(dim=-1)
        Y_true.extend(yb.numpy())
        Y_pred.extend(ypred.detach().cpu().numpy())
        V_ids.extend(video_ids)

    Y_true = np.array(Y_true)
    Y_pred = np.array(Y_pred)
    acc = (Y_true == Y_pred).sum() / len(Y_true)
    stats_db.update(V_ids, Y_pred, Y_true, str(cfg.EVAL.RESULT_DIR), 'eval', -1)
    precision, recall, f1, _ = precision_recall_fscore_support(Y_true, Y_pred, average='macro')
    cnf_mat = confusion_matrix(Y_true, Y_pred, labels=labels)
    stats['eval'] = {
        'acc': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cnf_mat.tolist(),
        'labels': labels,
        'iou_slow': np.mean(iou_metrics['iou_slow']),
        'iou_fast': np.mean(iou_metrics['iou_slow']),
        'dice_slow': np.mean(iou_metrics['iou_slow']),
        'dice_fast': np.mean(iou_metrics['iou_slow']),
    }

    return stats
