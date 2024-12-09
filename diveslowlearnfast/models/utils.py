from pathlib import Path

import torch

import os

from glob import glob

from diveslowlearnfast.config import Config


def save_checkpoint(model, optimiser, epoch: int, cfg: Config):
    if len(cfg.TRAIN.CHECKPOINT_FILENAME) > 0:
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
