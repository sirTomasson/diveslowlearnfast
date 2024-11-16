import torch
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

    for xb, yb in tqdm(loader):
        yb = yb.to(device)
        xb_fast = xb[:].to(device)
        # reduce the number of frames by the alpha ratio
        # B x C x T / alpha x H x W
        xb_slow = xb[:, :, ::cfg.SLOWFAST.ALPHA].to(device)

        optimiser.zero_grad()
        o = model([xb_slow, xb_fast])
        loss = criterion(o, yb)
        loss.backward()
