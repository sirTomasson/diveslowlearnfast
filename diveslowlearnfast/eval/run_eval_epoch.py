import torch

import numpy as np

from torch import nn, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from diveslowlearnfast.config import Config
from diveslowlearnfast.train import helper as train_helper

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


@torch.no_grad()
def run_eval_epoch(model: nn.Module,
                   loader: DataLoader,
                   device,
                   cfg: Config,
                   labels,
                   stats,
                   scaler: GradScaler = None):
    loader_iter = iter(loader)
    eval_bar = tqdm(range(len(loader)), desc='Eval batch')
    model.eval()
    Y_true = []
    Y_pred = []
    losses = []
    for _ in eval_bar:
        xb, yb, io_times, transform_times = next(loader_iter)
        yb = yb.to(device)
        o = train_helper.forward(model, xb, device, cfg, scaler)
        ypred = o.argmax(dim=-1)
        Y_true.extend(yb.detach().cpu().numpy())
        Y_pred.extend(ypred.detach().cpu().numpy())

    Y_true = np.array(Y_true)
    Y_pred = np.array(Y_pred)
    acc = (Y_true == Y_pred).sum() / len(Y_true)
    precision, recall, f1, _ = precision_recall_fscore_support(Y_true, Y_pred, average='macro')
    cnf_mat = confusion_matrix(Y_true, Y_pred, labels=labels)
    loss = np.mean(losses)
    stats['eval'] = {
        'acc': acc,
        'loss': loss,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cnf_mat.tolist(),
        'labels': labels
    }
    return stats
