from pathlib import Path

import torch

import os

from glob import glob


def save_checkpoint(model, optimiser, epoch: int, save_dir: str | Path):
    path = os.path.join(save_dir, f'checkpoint_{epoch:04d}.pth')
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


def last_checkpoint(save_dir: str | Path):
    checkpoints = glob(os.path.join(save_dir, 'checkpoint_*.pth'))
    if len(checkpoints) <= 0:
        return None

    checkpoints.sort()
    return checkpoints[-1]
