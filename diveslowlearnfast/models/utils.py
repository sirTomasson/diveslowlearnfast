from pathlib import Path

import torch

import os

from glob import glob

from diveslowlearnfast.config import Config


def save_checkpoint(model, optimiser, epoch: int, cfg: Config, filename=None):
    if filename:
        path = os.path.join(cfg.TRAIN.RESULT_DIR, filename)
    elif len(cfg.TRAIN.CHECKPOINT_FILENAME) > 0:
        path = os.path.join(cfg.TRAIN.RESULT_DIR, cfg.TRAIN.CHECKPOINT_FILENAME)
    else:
        path = os.path.join(cfg.TRAIN.RESULT_DIR, f'checkpoint_{epoch:04d}.pth')

    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimiser.state_dict(),
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, optimiser, checkpoint_path: str | Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    optimiser.load_state_dict(checkpoint['optimizer_state'])
    return model, optimiser, checkpoint['epoch']


def last_checkpoint(cfg: Config):
    if len(cfg.TRAIN.CHECKPOINT_FILENAME) > 0:
        checkpoint_path = os.path.join(cfg.TRAIN.RESULT_DIR, cfg.TRAIN.CHECKPOINT_FILENAME)
        if os.path.exists(checkpoint_path):
            return checkpoint_path

    checkpoints = glob(os.path.join(cfg.TRAIN.RESULT_DIR, 'checkpoint_*.pth'))
    if len(checkpoints) <= 0:
        return None

    checkpoints.sort()
    return checkpoints[-1]

def get_parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def to_slowfast_inputs(xb, alpha, requires_grad=False, device: torch.device=torch.device('cpu')):
    xb_fast = xb[:].to(device)
    # reduce the number of frames by the alpha ratio
    # B x C x T / alpha x H x W
    xb_slow = xb[:, :, ::alpha].to(device)
    xb_fast.requires_grad = requires_grad
    xb_slow.requires_grad = requires_grad

    return [xb_slow, xb_fast]
