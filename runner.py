import torch

import numpy as np

from datetime import datetime


from diveslowlearnfast.config import (
    merge_config,
    Config,
    parse_args
)

from diveslowlearnfast.datasets import Diving48Dataset
from diveslowlearnfast.models import SlowFast
from diveslowlearnfast.train import train_epoch

from pytorchvideo.transforms import ShortSideScale, Div255, UniformTemporalSubsample
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from diveslowlearnfast.transforms import CenterCropVideo, Permute, ToTensor4D

def print_device_props(device):
    print(f'Running on {device}')

    if device == torch.device('cpu'): return

    device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
    print(f"Device: {device_props.name}")
    print(f"Total memory: {device_props.total_memory / 1024 ** 2:.2f} MB")
    print(f"GPU number: {device_props.major}.{device_props.minor}")


def main():
    cfg = Config()
    args = parse_args()
    cfg = merge_config(cfg, args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_device_props(device)

    model = SlowFast(cfg).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(
        model.parameters(),
        lr=cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )

    transform = Compose([
        ToTensor4D(),
        Permute(3, 0, 1, 2),
        Div255(),  # Div255 assumes C x T x W x H
        ShortSideScale(size=256),
        # center crop
        CenterCropVideo(size=224),
        UniformTemporalSubsample(32)
        # ensures each sample has the same number of frames, will under sample if T < 128, and over sample if T > 128
    ])

    dataset = Diving48Dataset(
        cfg.DATA.VIDEOS_PATH,
        cfg.DATA.ANNOTATIONS_PATH,
        transform_fn=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        shuffle=True,
    )

    epoch_times = []
    for epoch in range(cfg.SOLVER.MAX_EPOCH):
        start = datetime.now()
        train_epoch(
            model,
            criterion,
            optimiser,
            dataloader,
            device,
            cfg
        )
        epoch_times.append((datetime.now() - start).seconds)
        mean_time = np.mean(epoch_times)
        print(f'Epoch {epoch + 1}/{cfg.SOLVER.MAX_EPOCH} - took {epoch_times[-1]:.3f} s, average time {mean_time:.3f} s')


if __name__ == '__main__':
    main()
