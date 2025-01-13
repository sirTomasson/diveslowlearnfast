import torch

import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from diveslowlearnfast.config import Config
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


@torch.no_grad()
def run_eval_epoch(model: nn.Module,
                   criterion: nn.Module,
                   loader: DataLoader,
                   device,
                   cfg: Config,
                   labels,
                   stats):
    loader_iter = iter(loader)
    eval_bar = tqdm(range(4), desc='Eval batch')
    model.eval()
    Y_true = []
    Y_pred = []
    losses = []
    for _ in eval_bar:
        xb, yb, io_times, transform_times = next(loader_iter)
        yb = yb.to(device)
        xb_fast = xb[:].to(device)
        # reduce the number of frames by the alpha ratio
        # B x C x T / alpha x H x W
        xb_slow = xb[:, :, ::cfg.SLOWFAST.ALPHA].to(device)

        o = model([xb_slow, xb_fast])
        loss = criterion(o, yb)
        ypred = o.argmax(dim=-1)
        Y_true.append(yb.detach().cpu().numpy())
        Y_pred.append(ypred.detach().cpu().numpy())
        losses.append(loss.item())

    Y_true = np.stack(Y_true).reshape(-1)
    Y_pred = np.stack(Y_pred).reshape(-1)
    print(Y_true[:5])
    print(Y_pred[:5])
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
        'confusion_matrix': cnf_mat.tolist()
    }
    return stats
